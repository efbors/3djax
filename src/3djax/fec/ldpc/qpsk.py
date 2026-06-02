""" qpsk.py

"""
import jax
import jax.numpy as jnp
import numpy as np


class QPSK:
    def __init__(self, phase_offset=jnp.pi / 8):
        """
        JAX-Optimized QPSK Modulator/Demodulator.
        Defaults match MATLAB's pskmod(..., 4, pi/8, 'InputType', 'Bit') with default 'Binary' mapping.
        """
        self.phase_offset = phase_offset

        # MATLAB default 'Binary' mapping for M=4:
        # '00' -> 0, '01' -> 1, '10' -> 2, '11' -> 3
        k = jnp.arange(4)
        phases = self.phase_offset + k * (jnp.pi / 2)

        # Create constellation points as complex numbers
        self.constellation = jnp.exp(1j * phases)

        # Bind JIT-compiled graphs here, capturing self.constellation in the closure!
        self.modulate = jax.jit(self._modulate)
        self.demodulate_approxllr = jax.jit(self._demodulate_approxllr)

    def _modulate(self, bits):
        """
        Maps a batch of bits to QPSK symbols using Gray Mapping (MATLAB default).
        """
        batch_size, n_bits = bits.shape
        bit_pairs = bits.reshape((batch_size, n_bits // 2, 2))

        b0 = bit_pairs[:, :, 0]
        b1 = bit_pairs[:, :, 1]

        # Gray mapping index calculation:
        # [0, 0] -> 0, [0, 1] -> 1, [1, 1] -> 2, [1, 0] -> 3
        indices = (b0 * 2) + jnp.bitwise_xor(b0, b1)

        symbols = self.constellation[indices]
        return symbols

    def _demodulate_approxllr(self, received_symbols, noise_var):
        """
        Calculates the approximate LLR for each bit using Gray Mapping.
        """
        y = jnp.expand_dims(received_symbols, axis=-1)
        c = jnp.expand_dims(self.constellation, axis=(0, 1))

        d2 = jnp.square(jnp.abs(y - c))

        # For bit 0 (MSB):
        # 0 is at indices 0 ('00') and 1 ('01')
        # 1 is at indices 2 ('11') and 3 ('10')
        min_d2_b0_is_0 = jnp.minimum(d2[:, :, 0], d2[:, :, 1])
        min_d2_b0_is_1 = jnp.minimum(d2[:, :, 2], d2[:, :, 3])
        llr_b0 = (min_d2_b0_is_1 - min_d2_b0_is_0) / noise_var

        # For bit 1 (LSB):
        # 0 is at indices 0 ('00') and 3 ('10')
        # 1 is at indices 1 ('01') and 2 ('11')
        min_d2_b1_is_0 = jnp.minimum(d2[:, :, 0], d2[:, :, 3])
        min_d2_b1_is_1 = jnp.minimum(d2[:, :, 1], d2[:, :, 2])
        llr_b1 = (min_d2_b1_is_1 - min_d2_b1_is_0) / noise_var

        llrs = jnp.stack([llr_b0, llr_b1], axis=-1)
        llrs_flat = llrs.reshape((received_symbols.shape[0], -1))

        return llrs_flat.astype(jnp.float16)


@jax.jit
def awgn_channel(symbols, snr_linear, key):
    """
    Adds complex Gaussian noise to symbols based on linear SNR.
    Assumes signal power is 1.0 (which it is for our phase-shifted QPSK).
    """
    noise_var = 1.0 / snr_linear

    # Split the key to generate independent real and imaginary noise
    key_r, key_i = jax.random.split(key)

    # Standard deviation per dimension (real/imaginary) is sqrt(noise_var / 2)
    std_dev = jnp.sqrt(noise_var / 2.0)

    noise_real = jax.random.normal(key_r, symbols.shape, dtype=jnp.float32) * std_dev
    noise_imag = jax.random.normal(key_i, symbols.shape, dtype=jnp.float32) * std_dev

    noise = noise_real + 1j * noise_imag
    return symbols + noise, noise_var


# ==============================================================================
# Uncoded BER vs SNR Simulation Harness
# ==============================================================================
if __name__ == "__main__":
    import time

    # Simulation parameters
    N = 1944  # Wi-Fi codeword length
    BATCH_SIZE = 500000  # Massive batch to saturate the RTX 3090
    TOTAL_BITS = N * BATCH_SIZE

    snr_db_range = np.arange(4, 7.7, 0.25)  # From very noisy to clean

    # Initialize Modulator
    qpsk = QPSK()

    # Initialize JAX random state
    key = jax.random.PRNGKey(42)

    print(f"Starting Uncoded QPSK Simulation.")
    print(f"Batch Size: {BATCH_SIZE} codewords per SNR ({TOTAL_BITS} bits total)")
    print("-" * 50)
    print(f"{'SNR (dB)':<10} | {'BER':<15} | {'Bit Errors':<15} | {'Time (s)':<10}")
    print("-" * 50)

    for snr_db in snr_db_range:
        start_time = time.time()

        # 1. Generate random bits
        key, bit_key, noise_key = jax.random.split(key, 3)
        bits = jax.random.randint(bit_key, shape=(BATCH_SIZE, N), minval=0, maxval=2, dtype=jnp.uint8)

        # 2. Modulate
        symbols = qpsk.modulate(bits)

        # 3. Add AWGN Noise
        snr_linear = 10.0 ** (snr_db / 10.0)
        noisy_symbols, noise_var = awgn_channel(symbols, snr_linear, noise_key)

        # 4. Demodulate (Get LLRs)
        llrs = qpsk.demodulate_approxllr(noisy_symbols, noise_var)

        # 5. Hard Slicer (LLR < 0 -> 1, LLR > 0 -> 0)
        rx_bits = jnp.where(llrs < 0, 1, 0).astype(jnp.uint8)

        # 6. Count Errors
        # We use np.asarray to pull the scalar result out of GPU VRAM into CPU RAM
        errors = np.asarray(jnp.sum(rx_bits != bits))
        ber = errors / TOTAL_BITS

        elapsed = time.time() - start_time

        # First iteration includes JIT compile time, so we note it.
        compile_note = "(includes JIT)" if snr_db == snr_db_range[0] else ""

        print(f"{snr_db:<10.1f} | {ber:<15.6e} | {errors:<15} | {elapsed:<6.4f} {compile_note}")
