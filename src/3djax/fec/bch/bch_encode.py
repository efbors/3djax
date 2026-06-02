import numpy as np


class BCHEncode:
    def __init__(self, K=1900, t=4):
        """
        Initializes the BCH Encoder over GF(2^11) for t=4 errors.
        K : Length of the payload (default 1900 for shortened 1944-bit block).
        """
        self.K = K
        self.t = t
        self.parity_len = self.t * 11  # 44 bits for t=4
        self.N = self.K + self.parity_len
        self.max_len = 2048  # Preallocation padding target

        # 1. Generate GF(2^11) Log/Exp Lookup Tables (For the Decoder later)
        # Primitive polynomial for GF(2^11) is x^11 + x^2 + 1 (Integer: 2053)
        self.exp_table = np.zeros(2048, dtype=np.int32)
        self.log_table = np.zeros(2048, dtype=np.int32)
        self._generate_gf_tables()

        # 2. Build the Parity Generator Matrix P for GF(2) matrix multiplication
        # Generator Polynomial Exponents for t=4
        self.g_taps = [0, 6, 7, 8, 9, 15, 17, 18, 19, 22, 23, 24, 25, 26, 27, 32, 35, 37, 40, 41]

        # We precalculate the P matrix once in the constructor.
        # Shape: (K, 44). We use uint16 to prevent overflow during the dot product later.
        self.P = np.zeros((self.K, self.parity_len), dtype=np.uint16)
        self._build_parity_matrix()
        a = 0

    def _generate_gf_tables(self):
        """Generates the Log and Exp lookup tables for GF(2^11)."""
        state = 1
        for i in range(2047):
            self.exp_table[i] = state
            self.log_table[state] = i
            state <<= 1
            if state & 2048:  # If the 11th bit overflows
                state ^= 2053  # XOR with primitive polynomial

        self.log_table[0] = -1  # log(0) is undefined, map to -1 safely

    def _build_parity_matrix(self):
        """
        Calculates the parity contribution of every bit position in O(K) time
        by running a single integer-based LFSR sequentially backwards.
        """
        # Convert our g_taps array into a single 44-bit integer
        g_int = sum(1 << t for t in self.g_taps)

        # The '1' at the very end of the payload (index K-1) sits in the LFSR for exactly 1 cycle.
        # Its resulting state is exactly the generator taps.
        current_state = g_int

        # We iterate backwards from the last payload bit (K-1) up to the first (0)
        for i in range(self.K - 1, -1, -1):

            # Extract the 44 bits from the integer into our uint16 row
            row = np.array([(current_state >> j) & 1 for j in range(44)], dtype=np.uint16)

            # Store reversed to match standard MSB-first transmission
            self.P[i] = row[::-1]

            # Evolve the LFSR for one clock cycle (equivalent to feeding a '0')
            # Check the feedback bit (which is the bit at index 43)
            feedback = (current_state >> 43) & 1

            # Shift left by 1 and mask to ensure it stays exactly 44 bits
            current_state = (current_state << 1) & ((1 << 44) - 1)

            # Apply the XOR gates if feedback was 1
            if feedback:
                current_state ^= g_int

    def __call__(self, batch_payload):
        """
        Executes the batch encoder using pure NumPy array operations.
        batch_payload: numpy array of shape (B, K), dtype uint8.
        Returns: numpy array of shape (B, 2048), zero-padded.
        """
        B = batch_payload.shape[0]

        # 1. Calculate Parity via Matrix Multiplication
        # We dot-product the (B, K) batch with the (K, 44) P matrix.
        # Note: We must cast payload to uint16 so the dot product sum doesn't 
        # overflow uint8's max value of 255 (since K=1900).
        # Modulo 2 (& 1) turns the integer sums back into GF(2) XOR logic.
        batch_parity = np.bitwise_and(
            np.dot(batch_payload.astype(np.uint16), self.P), 1
        ).astype(np.uint8)

        # 2. Preallocate the (B, 2048) output buffer filled with zeros
        out_batch = np.zeros((B, self.max_len), dtype=np.uint8)

        # 3. Paste the Payload and Parity into the buffer (Systematic Encoding)
        out_batch[:, :self.K] = batch_payload
        out_batch[:, self.K: self.N] = batch_parity

        # The remaining indices (from self.N to 2048) naturally remain 0
        return out_batch