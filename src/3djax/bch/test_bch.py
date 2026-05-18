import numpy as np
import galois
import time
from bch_encode import BCHEncode


def main_bch():
    K = 1900  # shortened payload size
    t = 4  # number of corrections
    batch_size = 10000

    # Initialize our Custom NumPy Encoder
    print("Initializing Custom Encoder...")
    bch_encode_custom = BCHEncode(K, t)

    # 2. Initialize the Reference Galois Encoder
    # GF(2^11) max length is 2047. 
    # t=4 means 44 parity bits. Natural payload length is 2047 - 44 = 2003.
    print("Initializing Galois Reference Encoder...")
    bch_galois = galois.BCH(2047, 2003)

    # 3. Generate a batch of 10 random payloads (Shape: 10 x 1900)
    np.random.seed(42)
    batch_payload = np.random.randint(0, 2, size=(batch_size, K), dtype=np.uint8)

    print("\n--- Running Custom NumPy Encoder ---")
    # Custom encoder returns (10, 2048). The valid codeword is the first 1944 bits.
    custom_out = bch_encode_custom(batch_payload)
    custom_codewords = custom_out[:, :1944]

    print("--- Running Galois Reference Encoder ---")
    # Galois mathematically requires the natural payload length of 2003.
    # We "shorten" the code by padding 103 zeros to the MSB (the front).
    padding = np.zeros((batch_size, 103), dtype=np.uint8)
    padded_payload = np.concatenate((padding, batch_payload), axis=1)

    # ==========================================
    # TIME CUSTOM NUMPY ENCODER
    # ==========================================
    t0 = time.perf_counter()

    custom_out = bch_encode_custom(batch_payload)
    custom_codewords = custom_out[:, :1944]

    t1 = time.perf_counter()
    custom_time = t1 - t0

    # ==========================================
    # TIME GALOIS REFERENCE ENCODER
    # ==========================================
    t2 = time.perf_counter()

    gf_payload = galois.GF2(padded_payload)
    galois_out = bch_galois.encode(gf_payload)
    galois_codewords = np.asarray(galois_out)[:, 103:]

    t3 = time.perf_counter()
    galois_time = t3 - t2


    # Verification
    print("\n=== VERIFICATION & TIMING ===")
    matches = np.array_equal(custom_codewords, galois_codewords)

    if matches:
        print("✅ SUCCESS: Encoders match perfectly.")
    else:
        print("❌ MISMATCH: Outputs are different.")

    print(f"\nCustom NumPy Time: {custom_time:.4f} seconds")
    print(f"Galois Library Time: {galois_time:.4f} seconds")

    if custom_time < galois_time:
        speedup = galois_time / custom_time


if __name__ == '__main__':
    main_bch()