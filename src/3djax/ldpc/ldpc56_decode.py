import jax
import jax.numpy as jnp
import numpy as np


class LDPC56Decoder:
    def __init__(self, Z=81, max_iter=10, algo='oms', offset=0.25, scaling_factor=0.75):
        """
        XLA-Optimized Layered LDPC Decoder (IEEE 802.11 N=1944, Z=81, Rate 5/6).
        Supports NMS ('nms') and OMS ('oms').
        """
        self.Z = Z
        self.max_iter = max_iter
        self.algo = algo.lower()
        self.offset = offset
        self.scaling_factor = scaling_factor

        # Base Parity-Check Matrix Hb
        Hb = np.array([
            [13, 48, 80, 66, 4, 74, 7, 30, 76, 52, 37, 60, -1, 49, 73, 31, 74, 73, 23, -1, 1, 0, -1, -1],
            [69, 63, 74, 56, 64, 77, 57, 65, 6, 16, 51, -1, 64, -1, 68, 9, 48, 62, 54, 27, -1, 0, 0, -1],
            [51, 15, 0, 80, 24, 25, 42, 54, 44, 71, 71, 9, 67, 35, -1, 58, -1, 29, -1, 53, 0, -1, 0, 0],
            [16, 29, 36, 41, 44, 56, 59, 37, 50, 24, -1, 65, 4, 65, 52, -1, 4, -1, 73, 52, 1, -1, -1, 0]
        ], dtype=np.int32)

        # Precompute arrays
        cn_idx = np.zeros((4, 24, self.Z), dtype=np.int32)
        vn_idx = np.zeros((4, 24, self.Z), dtype=np.int32)

        # New static tensors for the math tricks
        mag_bias = np.zeros((4, 1, 24, 1), dtype=np.float32)
        r_mult = np.ones((4, 1, 24, 1), dtype=np.float32)
        is_conn = np.ones((4, 1, 24, 1), dtype=bool)

        for r in range(4):
            for c in range(24):
                shift = Hb[r, c]
                if shift != -1:
                    cn_idx[r, c, :] = (np.arange(self.Z) - shift) % self.Z
                    vn_idx[r, c, :] = (np.arange(self.Z) + shift) % self.Z
                else:
                    # Identity pass-through for disconnected edges
                    cn_idx[r, c, :] = np.arange(self.Z)
                    vn_idx[r, c, :] = np.arange(self.Z)

                    # Force to lose min-search
                    mag_bias[r, 0, c, 0] = np.inf
                    # Force check message to exactly 0.0
                    r_mult[r, 0, c, 0] = 0.0
                    # Force sign to +1.0
                    is_conn[r, 0, c, 0] = False

        self.cn_idx = jnp.array(cn_idx)
        self.vn_idx = jnp.array(vn_idx)
        self.mag_bias = jnp.array(mag_bias)
        self.r_mult = jnp.array(r_mult)
        self.is_conn = jnp.array(is_conn)
        self.col_indices = jnp.arange(24).reshape(1, 24, 1)


        # Bind the pure function, compiling the static arrays into the graph
        self.decode = jax.jit(self._decode_static)

    def _decode_static(self, llrs):
        """
        llrs: float16 array of shape (Batch, 24, 81)
        Returns: decoded info bits of shape (Batch, 20, 81), dtype=uint8
        """
        batch_size = llrs.shape[0]

        # Initialize state: L (Total LLRs), R (Check Node Messages)
        L = llrs
        R = jnp.zeros((4, batch_size, 24, self.Z), dtype=jnp.float16)

        def bp_iteration(state, _):
            L_curr, R_curr = state

            # Upcast to float32 for mathematically stable local accumulations
            L_curr = L_curr.astype(jnp.float32)
            R_curr = R_curr.astype(jnp.float32)

            # Python loop entirely unrolls the 4 layers during JIT compilation
            # Inside _decode_static -> bp_iteration:
            for r in range(4):
                #  Unconditional Gather
                L_shifted = L_curr[:, jnp.arange(24)[:, None], self.cn_idx[r]]
                R_old = R_curr[r]

                #  Variable to Check Message (Q)
                Q = L_shifted - R_old

                # 3. Pure Tensor Min-Sum
                # Add infinity to disconnected edges so they lose the min search
                mag = jnp.abs(Q) + self.mag_bias[r]

                # Combined sign logic (Disconnected edges evaluate to False -> +1.0)
                sign = jnp.where((Q < 0) & self.is_conn[r], -1.0, 1.0)

                prod_sign = jnp.prod(sign, axis=1, keepdims=True)
                excl_sign = prod_sign * sign

                idx1 = jnp.argmin(mag, axis=1, keepdims=True)
                m1 = jnp.min(mag, axis=1, keepdims=True)

                mag_no_m1 = jnp.where(self.col_indices == idx1, jnp.inf, mag)
                m2 = jnp.min(mag_no_m1, axis=1, keepdims=True)

                out_mag = jnp.where(self.col_indices == idx1, m2, m1)

                if self.algo == 'oms':
                    out_mag = jnp.maximum(out_mag - self.offset, 0.0)
                elif self.algo == 'nms':
                    out_mag = out_mag * self.scaling_factor

                # Nullify R_new for disconnected edges via simple multiplication
                R_new = excl_sign * out_mag * self.r_mult[r]

                # Check to Variable Update
                # Disconnected edges: L_shifted_new = Q + 0.0 = L_shifted
                L_shifted_new = Q + R_new

                # 6. Unconditional Scatter
                # Disconnected edges write their unmodified value back to its original location
                L_unshifted = L_shifted_new[:, jnp.arange(24)[:, None], self.vn_idx[r]]
                L_curr = L_unshifted

                R_curr = R_curr.at[r].set(R_new)

            # Downcast back to float16 to preserve VRAM bandwidth between iterations
            return (L_curr.astype(jnp.float16), R_curr.astype(jnp.float16)), None

        # Execute the BP iterations via jax.lax.scan (keeps the graph extremely lightweight)
        (L_final, _), _ = jax.lax.scan(bp_iteration, (L, R), None, length=self.max_iter)

        # Hard Slicer on Information Bits only (first 20 columns)
        # LLR < 0 -> Bit 1, LLR >= 0 -> Bit 0
        info_L = L_final[:, :20, :]
        rx_bits = jnp.where(info_L < 0, 1, 0).astype(jnp.uint8)

        return rx_bits