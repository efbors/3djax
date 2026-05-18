"""
test_ldpc.py

"""
import time
import datetime
import numpy as np
import jax
import jax.numpy as jnp

from ldpc56_encode import LDPC56Encode
from ldpc56_decode import LDPC56Decoder
from qpsk import QPSK, awgn_channel

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Simulation Parameters (Matching MATLAB)
    # -------------------------------------------------------------------------
    # snr_start = 4.25
    # snr_end = snr_start + 3.25
    # snr_step = 0.25

    snr_start = 6.75
    snr_end = snr_start + 1.
    snr_step = 0.25

    snr_range = np.arange(snr_start, snr_end + snr_step / 2, snr_step)

    Z = 81
    max_iter = 12
    total_blocks = 50_000_000
    chunk_size = 200_000
    num_chunks = total_blocks // chunk_size

    offset_val = 0.4
    scaling_val = 0.75

    info_bits_per_block = 20 * Z
    bits_per_chunk = chunk_size * info_bits_per_block

    # -------------------------------------------------------------------------
    # Initialize JAX Modules
    # -------------------------------------------------------------------------
    print("Initializing modules and building static XLA graphs...")
    encoder = LDPC56Encode(Z=Z)
    qpsk = QPSK()

    # Instantiate both NMS and OMS decoders to match the MATLAB parallel run
    decoder_nms = LDPC56Decoder(Z=Z, max_iter=max_iter, algo='nms', scaling_factor=scaling_val)
    decoder_oms = LDPC56Decoder(Z=Z, max_iter=max_iter, algo='oms', offset=offset_val)


    # -------------------------------------------------------------------------
    # End-to-End JIT Compiled Chunk Processor
    # -------------------------------------------------------------------------
    @jax.jit
    def process_chunk(key, snr_linear):
        key_payload, key_noise = jax.random.split(key)

        # 1. Generate Info Payload: (Batch, 20, 81)
        payload = jax.random.randint(key_payload, shape=(chunk_size, 20, Z), minval=0, maxval=2, dtype=jnp.uint8)

        # 2. Encode to Codeword: (Batch, 24, 81)
        codeword = encoder.encode(payload)

        # 3. Modulate
        # Flatten the 24x81 layout for the modulator which expects (Batch, N)
        bits_flat = codeword.reshape((chunk_size, 24 * Z))
        symbols = qpsk.modulate(bits_flat)

        # 4. AWGN Channel
        noisy_symbols, noise_var = awgn_channel(symbols, snr_linear, key_noise)

        # 5. Demodulate LLRs
        llrs_flat = qpsk.demodulate_approxllr(noisy_symbols, noise_var)
        # Reshape back to the 2D layout for the decoder
        llrs = llrs_flat.reshape((chunk_size, 24, Z))

        # 6. Decode
        rx_bits_nms = decoder_nms.decode(llrs)
        rx_bits_oms = decoder_oms.decode(llrs)

        # 7. Uncoded (Raw) Bit Decisions
        # Extract the information LLRs (first 20 blocks) and slice at 0
        info_llrs = llrs[:, :20, :]
        rx_bits_raw = jnp.where(info_llrs < 0, 1, 0).astype(jnp.uint8)

        # 8. Count Errors
        err_raw = jnp.sum(rx_bits_raw != payload)
        err_nms = jnp.sum(rx_bits_nms != payload)
        err_oms = jnp.sum(rx_bits_oms != payload)

        return err_raw, err_nms, err_oms


    # -------------------------------------------------------------------------
    # Simulation Loop
    # -------------------------------------------------------------------------
    log_filename = 'ldpc_simulation_log_jax.txt'
    with open(log_filename, 'w') as fid:
        fid.write(f"LDPC JAX Simulation Log - {datetime.datetime.now()}\n")
        fid.write("CNK  | SNR(dB) | Raw_BER     | NMS_BER     | OMS_BER\n")
        fid.write("-------------------------------------------------------\n")

    print("\nStarting Simulation...")
    print(f"{'SNR(dB)':<8} | {'Chunks':<6} | {'Raw BER':<11} | {'NMS BER':<11} | {'OMS BER':<11} | {'Time(s)'}")
    print("-" * 70)

    key = jax.random.PRNGKey(0)  # Equivalent to RndSeed = 0
    global_start_time = time.time()

    # Trigger JIT compilation before the loop so timing is accurate
    print("Compiling JIT function (this takes a few seconds)...")
    _ = process_chunk(key, 10 ** (snr_start / 10.0))
    print("JIT compilation finished. Running loop.\n")

    # Final Result Arrays
    ber_raw_vec = np.zeros(len(snr_range))
    ber_nms_vec = np.zeros(len(snr_range))
    ber_oms_vec = np.zeros(len(snr_range))

    for i, snr_val in enumerate(snr_range):
        snr_linear = 10.0 ** (snr_val / 10.0)

        acc_err_raw = 0
        acc_err_nms = 0
        acc_err_oms = 0
        acc_bits = 0

        snr_start_time = time.time()

        for c in range(1, num_chunks + 1):
            key, iter_key = jax.random.split(key)

            # Run the chunk on GPU
            err_raw, err_nms, err_oms = process_chunk(iter_key, snr_linear)

            # Pull scalars to CPU for accumulation and logic
            acc_err_raw += int(np.asarray(err_raw))
            acc_err_nms += int(np.asarray(err_nms))
            acc_err_oms += int(np.asarray(err_oms))
            acc_bits += bits_per_chunk

            # Calculate running BERs
            ber_raw = acc_err_raw / acc_bits
            ber_nms = acc_err_nms / acc_bits
            ber_oms = acc_err_oms / acc_bits

            # Log to File
            with open(log_filename, 'a') as fid:
                fid.write(f"{c:<4} | {snr_val:<7.2f} | {ber_raw:<11.4e} | {ber_nms:<11.4e} | {ber_oms:<11.4e}\n")

            # Early Exit Check
            if acc_err_oms > 1000 and acc_err_nms > 1000:
                break

        # Store final BER for this SNR
        ber_raw_vec[i] = ber_raw
        ber_nms_vec[i] = ber_nms
        ber_oms_vec[i] = ber_oms

        elapsed = time.time() - snr_start_time
        print(f"{snr_val:<8.2f} | {c:<6} | {ber_raw:<11.4e} | {ber_nms:<11.4e} | {ber_oms:<11.4e} | {elapsed:.2f}")

    total_time = time.time() - global_start_time
    print("-" * 70)
    print(f"Simulation completed in {total_time / 60:.2f} minutes.")
    print("Results saved to:", log_filename)
