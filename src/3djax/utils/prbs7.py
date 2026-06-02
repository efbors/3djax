

def prbs7(seed=0x7F):
    """
    Generate one full PRBS7 period (127 bits)
    Polynomial: x^7 + x^6 + 1
    Seed must be nonzero and fit in 7 bits.
    """
    if seed == 0 or seed >= 0x80:
        raise ValueError("Seed must be 1..127")

    state = seed
    bits = []

    for _ in range(127):
        out = state & 1
        bits.append(out)

        # Feedback from bit0 and bit1 (corresponding to x^7 + x^6 + 1)
        fb = ((state >> 0) ^ (state >> 1)) & 1

        # Shift right and insert feedback into MSB (bit 6)
        state = (state >> 1) | (fb << 6)

    return bits
