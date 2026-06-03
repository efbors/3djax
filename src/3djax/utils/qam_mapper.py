import numpy as np

def get_qam_lut(modulation='QAM16'):
    """Returns the 1D Gray-coded mapping and normalization factor for Wi-Fi standard QAM."""
    if modulation == 'QAM4':
        # 1 bit per channel: 0->-1, 1->+1
        lut = np.array([-1, 1], dtype=float)
        k_mod = 1.0 / np.sqrt(2)
        bits_per_ch = 1
    elif modulation == 'QAM16':
        # 2 bits per channel: 00->-3, 01->-1, 11->1, 10->3
        lut = np.array([-3, -1, 3, 1], dtype=float)
        k_mod = 1.0 / np.sqrt(10)
        bits_per_ch = 2
    elif modulation == 'QAM64':
        # 3 bits per channel: Wi-Fi standard 8-level mapping
        lut = np.array([-7, -5, -1, -3, 7, 5, 1, 3], dtype=float)
        k_mod = 1.0 / np.sqrt(42)
        bits_per_ch = 3
    else:
        raise ValueError("Unsupported modulation")
    return lut, k_mod, bits_per_ch
