from scipy.signal import bessel, lfilter
import numpy as np


def generate_bessel_taps(cutoff_ratio, span):
    """
    Generates the FIR impulse response of a 4th-order Bessel-Thomson low-pass filter.
    cutoff_ratio : Nyquist = 1., Fs = 2.
    span         : Length of the generated impulse response in samples
    """

    # -- Generate digital 4th-order Bessel filter coefficients (IIR)
    N_order = 4
    b, a = bessel(N=N_order, Wn=cutoff_ratio, btype='low', analog=False, output='ba')  # return the IIR filter

    #  Create a discrete impulse to excite the filter
    impulse = np.zeros(span, dtype=np.float32)
    impulse[0] = 1.0

    #  Filter the impulse to extract the impulse response (FIR approximation)
    taps = lfilter(b, a, impulse).astype(np.float32)

    #  Align the phase for fft_convolve_same
    # A physical Bessel filter is causal (the energy comes after t=0).
    # fft_convolve_same trims the array assuming the filter is centered.
    # To prevent the waveform from shifting backward in time, we roll the
    # peak of the Bessel response to exactly the center of the tap array.
    peak_idx = np.argmax(taps)
    center_idx = (span - 1) // 2
    taps = np.roll(taps, center_idx - peak_idx)

    if False:
        import matplotlib.pyplot as plt
        plt.plot(taps, 'r')
        plt.grid()
        plt.show(block=False)

    # -- Normalize to ensure exactly 0 dB DC gain
    return taps / np.sum(taps)
