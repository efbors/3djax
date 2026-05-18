import numpy as np
import jax
import jax.numpy as jnp


class LDPC56Encode:
    def __init__(self, Z=81):
        """
        XLA-Optimized LDPC Encoder for IEEE 802.11 N=1944, Z=81, Rate 5/6.
        All permutation and masking logic is resolved at instantiation to guarantee
        a static graph for the JIT compiler.
        """
        self.Z = Z
        self.Mb = 4
        self.Nb = 24
        self.info_blocks = self.Nb - self.Mb

        # Base Parity-Check Matrix Hb
        Hb = np.array([
            [13, 48, 80, 66, 4, 74, 7, 30, 76, 52, 37, 60, -1, 49, 73, 31, 74, 73, 23, -1, 1, 0, -1, -1],
            [69, 63, 74, 56, 64, 77, 57, 65, 6, 16, 51, -1, 64, -1, 68, 9, 48, 62, 54, 27, -1, 0, 0, -1],
            [51, 15, 0, 80, 24, 25, 42, 54, 44, 71, 71, 9, 67, 35, -1, 58, -1, 29, -1, 53, 0, -1, 0, 0],
            [16, 29, 36, 41, 44, 56, 59, 37, 50, 24, -1, 65, 4, 65, 52, -1, 4, -1, 73, 52, 1, -1, -1, 0],
        ], dtype=np.int32)

        # Pre-compute bit-level maps
        # Define explicitly where every bit comes from to avoid runtime jnp.roll
        # Shape: (4 rows, 20 blocks, 81 bits)
        block_idx = np.zeros((self.Mb, self.info_blocks, self.Z), dtype=np.int32)
        bit_idx = np.zeros((self.Mb, self.info_blocks, self.Z), dtype=np.int32)
        mask = np.zeros((self.Mb, self.info_blocks, self.Z), dtype=np.uint8)

        for r in range(self.Mb):
            for c in range(self.info_blocks):
                shift = Hb[r, c]
                if shift != -1:
                    # Mark block index
                    block_idx[r, c, :] = c
                    # A cyclic right shift means destination bit 'i' comes from source bit '(i - shift) % Z'
                    bit_idx[r, c, :] = (np.arange(self.Z) - shift) % self.Z
                    mask[r, c, :] = 1
                else:
                    # Dummy indices to prevent out-of-bounds (will be zeroed by mask anyway)
                    block_idx[r, c, :] = 0
                    bit_idx[r, c, :] = 0
                    mask[r, c, :] = 0

        # Upload static arrays to JAX device memory
        self.block_idx = jnp.array(block_idx)
        self.bit_idx = jnp.array(bit_idx)
        self.mask = jnp.array(mask)

        # Extract the single parity rotation value needed for p1 (Hb[0, 20] which is 1)
        self.p0_shift = int(Hb[0, 20])

        # 2. BIND JIT-COMPILED GRAPH
        # We compile the function during init, capturing the static JAX arrays in the closure.
        self.encode = jax.jit(self._build_static_graph)

    def _build_static_graph(self, payload):
        """
        Pure JAX function with no branches or loops.
        payload shape: (Batch, 20, 81), dtype: uint8
        """
        # PHASE 1: Information Gather-Mask-Reduce
        # 1. Massive parallel index lookup -> Shape: (Batch, 4, 20, 81)
        gathered = payload[:, self.block_idx, self.bit_idx]

        # 2. Branchless mask (zeroes out the '-1' blocks)
        masked = gathered * self.mask

        # 3. Sum reduction across the 20 columns to get v0, v1, v2, v3
        # Max sum is ~18, so uint8 will not overflow. Shape: (Batch, 4, 81)
        v = jnp.sum(masked, axis=2, dtype=jnp.uint8)

        # PHASE 2: Parity Cascade
        # p0: Sum of all 4 rows
        p0 = (v[:, 0] + v[:, 1] + v[:, 2] + v[:, 3]) & 1

        # p1: Requires right-shifting p0
        p0_shifted = jnp.roll(p0, self.p0_shift, axis=-1)
        p1 = (v[:, 0] + p0_shifted) & 1

        # p2
        p2 = (v[:, 1] + p1) & 1

        # p3
        p3 = (v[:, 2] + p0 + p2) & 1

        # Stack parity blocks. Shape: (Batch, 4, 81)
        p = jnp.stack([p0, p1, p2, p3], axis=1)

        # Concatenate payload and parity to form the final systematic codeword
        # Output Shape: (Batch, 24, 81)
        codeword = jnp.concatenate([payload, p], axis=1)

        return codeword
