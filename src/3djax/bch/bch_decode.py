import numpy as np


class BCHDecode:
    def __init__(self, K=1900, t=4):
        """
        Initializes the Vectorized BCH Decoder over GF(2^11) for t=4 errors.
        """
        self.K = K
        self.t = t
        self.parity_len = self.t * 11
        self.N = 2047  # Natural length of GF(2^11)
        self.short_N = self.K + self.parity_len  # 1944
        self.pad_len = self.N - self.short_N  # 103

        # 1. Generate GF(2^11) Lookup Tables
        self.exp_table = np.zeros(2048, dtype=np.int32)
        self.log_table = np.zeros(2048, dtype=np.int32)
        self._generate_gf_tables()

        # 2. Precompute Chien Search Evaluation Roots
        # We only need to search the 1944 bits of our shortened frame.
        # Bit index i corresponds to polynomial degree (2046 - i).
        # The root to check is alpha^(-degree).
        eval_indices = (- (2046 - np.arange(self.pad_len, self.N))) % 2047
        self.chien_roots = self.exp_table[eval_indices]

    def _generate_gf_tables(self):
        state = 1
        for i in range(2047):
            self.exp_table[i] = state
            self.log_table[state] = i
            state <<= 1
            if state & 2048:
                state ^= 2053
        self.log_table[0] = -1

    def _gf_mul(self, x, y):
        """Vectorized GF(2^11) Multiplication."""
        mask = (x != 0) & (y != 0)
        res = np.zeros(np.broadcast(x, y).shape, dtype=np.int32)

        # Broadcast arrays to match output shape
        bx = np.broadcast_to(x, res.shape)
        by = np.broadcast_to(y, res.shape)
        bmask = np.broadcast_to(mask, res.shape)

        # Multiply using Log/Exp tables
        idx = (self.log_table[bx[bmask]] + self.log_table[by[bmask]]) % 2047
        res[bmask] = self.exp_table[idx]
        return res

    def __call__(self, batch_rx):
        """
        Executes the batch decoder.
        batch_rx: numpy array of shape (B, 1944), dtype uint8.
        Returns: numpy array of shape (B, 1900) containing the corrected payload.
        """
        B = batch_rx.shape[0]

        # 1. PAD TO NATURAL LENGTH
        # Shape becomes (B, 2047)
        padding = np.zeros((B, self.pad_len), dtype=np.int32)
        R = np.concatenate((padding, batch_rx), axis=1)

        # ==========================================
        # STEP 1: VECTORIZED SYNDROME CALCULATION
        # ==========================================
        S = np.zeros((B, 2 * self.t), dtype=np.int32)
        for k in range(1, 2 * self.t + 1):
            alpha_k = self.exp_table[k]
            S_k = np.zeros(B, dtype=np.int32)
            # Horner's method evaluated across the batch
            for i in range(self.N):
                S_k = self._gf_mul(S_k, alpha_k) ^ R[:, i]
            S[:, k - 1] = S_k

        # ==========================================
        # STEP 2: REFORMULATED INVERSIONLESS BM (RiBM)
        # ==========================================
        # Initialize polynomials for the batch
        Lambda = np.zeros((B, self.t + 1), dtype=np.int32)
        Lambda[:, 0] = 1

        B_poly = np.zeros((B, self.t + 1), dtype=np.int32)
        B_poly[:, 1] = 1  # Already shifted by 1

        gamma = np.ones(B, dtype=np.int32)
        L = np.zeros(B, dtype=np.int32)

        for k in range(2 * self.t):
            # Calculate Discrepancy (Delta)
            Delta = np.zeros(B, dtype=np.int32)
            for j in range(self.t + 1):
                if k - j >= 0:
                    Delta ^= self._gf_mul(Lambda[:, j], S[:, k - j])

            # Evaluate Branch Conditions (The Masks)
            cond = (Delta != 0) & (2 * L <= k)
            cond_mask = cond[:, None]  # Reshape for broadcasting

            # Calculate Next States
            Lambda_next = self._gf_mul(gamma[:, None], Lambda) ^ self._gf_mul(Delta[:, None], B_poly)
            B_poly_next = np.where(cond_mask, Lambda, B_poly)

            # Shift B_poly right by 1
            B_poly_next[:, 1:] = B_poly_next[:, :-1].copy()
            B_poly_next[:, 0] = 0

            # Update Registers
            Lambda = Lambda_next
            B_poly = B_poly_next
            L = np.where(cond, k + 1 - L, L)
            gamma = np.where(cond, Delta, gamma)

        # ==========================================
        # STEP 3: BATCH CHIEN SEARCH
        # ==========================================
        # Evaluate Lambda at all 1944 roots simultaneously
        result = Lambda[:, 0:1].copy()  # Lambda_0 is gamma in RiBM

        for j in range(1, self.t + 1):
            # Calculate (root)^j
            root_pow = self.exp_table[(self.log_table[self.chien_roots] * j) % 2047]
            # GF multiply Lambda_j * (root)^j
            term = self._gf_mul(Lambda[:, j:j + 1], root_pow[None, :])
            result ^= term

        # Where result == 0, we found an error!
        error_mask = (result == 0).astype(np.uint8)

        # ==========================================
        # STEP 4: CORRECT AND EXTRACT
        # ==========================================
        # Flip the bits in the shortened 1944-bit frame
        corrected_frame = batch_rx ^ error_mask

        # Extract the 1900-bit payload
        return corrected_frame[:, :self.K]