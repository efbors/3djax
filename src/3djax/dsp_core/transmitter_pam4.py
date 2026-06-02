import numpy as np
from scipy.signal import bessel, lfilter
from utils.prbs7 import prbs7

def generate_bessel_taps(os_factor, cutoff_ratio, span):
    """
    Generates the FIR impulse response of a 4th-order Bessel-Thomson low-pass filter.

    Parameters:
    os_factor    : Oversampling factor (e.g., 8 for 800 GHz on 100 Gbaud)
    cutoff_ratio : -3dB cutoff frequency relative to the baud rate.
                   (0.75 represents 75 GHz for a 100 Gbaud signal)
    span         : Length of the generated impulse response in symbols.
    """
    # 1. Calculate the digital normalized cutoff frequency (Wn)
    # Nyquist frequency is 0.5 * sample_rate.
    # sample_rate = os_factor * baud_rate.
    # Wn = (cutoff_ratio * baud_rate) / (0.5 * os_factor * baud_rate)
    Wn = (2.0 * cutoff_ratio) / os_factor

    # 2. Generate digital 4th-order Bessel filter coefficients (IIR)
    b, a = bessel(4, Wn, btype='low', analog=False, output='ba')

    # 3. Create a perfect discrete impulse to excite the filter
    tap_len = span * os_factor
    impulse = np.zeros(tap_len, dtype=np.float32)
    impulse[0] = 1.0

    # 4. Filter the impulse to extract the impulse response (FIR approximation)
    taps = lfilter(b, a, impulse).astype(np.float32)

    # 5. Align the phase for fft_convolve_same
    # A physical Bessel filter is causal (the energy comes after t=0).
    # fft_convolve_same trims the array assuming the filter is centered.
    # To prevent your waveform from shifting backward in time, we roll the
    # peak of the Bessel response to exactly the center of the tap array.
    peak_idx = np.argmax(taps)
    center_idx = (tap_len - 1) // 2
    taps = np.roll(taps, center_idx - peak_idx)

    # 6. Normalize to ensure exactly 0 dB DC gain
    return taps / np.sum(taps)


def generate_rrc_taps(os_factor, beta=0.35, span=8):
    """
    Generates Root-Raised-Cosine (RRC) filter taps using the epsilon trick
    to avoid divide-by-zero singularities.
    """
    # Epsilon trick avoids conditional branching for exactly t=0 and t=T/(4*beta)
    t = (np.arange(-span * os_factor, span * os_factor + 1) + 1e-12) / os_factor

    num = np.sin(np.pi * t * (1 - beta)) + 4 * beta * t * np.cos(np.pi * t * (1 + beta))
    den = np.pi * t * (1 - (4 * beta * t) ** 2)

    taps = num / den

    # Normalize to ensure 0 dB DC gain
    return (taps / np.sum(taps)).astype(np.float32)


def fft_convolve_same(signal, filter_taps):
    """
    Convolves a batched signal with filter_taps along the last dimension using FFT.
    Returns the 'same' length output.
    """
    sig_len = signal.shape[-1]
    tap_len = filter_taps.shape[-1]

    # Linear convolution requires padding to N+M-1
    full_len = sig_len + tap_len - 1

    # JAX/XLA optimal padding: Exact next power of two
    fast_len = 1 << (full_len - 1).bit_length()

    # Execute pure tensor FFTs along the last axis
    SIG = np.fft.rfft(signal, n=fast_len, axis=-1)
    TAPS = np.fft.rfft(filter_taps, n=fast_len, axis=-1)

    # Point-wise multiplication and Inverse FFT
    out = np.fft.irfft(SIG * TAPS, n=fast_len, axis=-1)

    # Trim the padding to return 'same' mode length
    start = (tap_len - 1) // 2
    end = start + sig_len

    return out[..., start:end]


class TransmitterPAM4:
    def __init__(self, config):
        self.config = config
        self.baud_rate = float(config['system']['baud_rate'])
        self.simulation_duration = config['system']['target_simulation_duration']
        self.start_delay = config['tx']['start_delay']
        self.os_factor = int(config['system']['os_factor'])

    def generate_signal(self):
        """
        Generate batched PRBS7Q-based sync symbols followed by random PAM4 symbols,
        upsample via zero-insertion, apply a smooth start ramp, insert physical start delay,
        and apply FFT-based Root-Raised-Cosine pulse shaping.
        """

        # The tx waveform does not start on sample zero;  there is a delay specified by start_delay_s
        # this delay is partitioned into two: an integer number of analog samples and a fractional one
        sample_rate = self.baud_rate * self.os_factor  # e.g. 100e9 * 8 = 800e9 samples/s
        sample_period = 1.0 / sample_rate  # e.g. 1.25 ps/sample
        symbol_period = 1.0 / self.baud_rate

        # Convert the desired start delay into samples (may be fractional)
        delay_samples = self.start_delay / sample_period
        integer_delay_samples = int(np.floor(delay_samples))

        # Calculate exact number of symbols needed to fill the duration
        num_symbols = int(np.ceil(self.simulation_duration / symbol_period)) + int(
            np.ceil(delay_samples / self.os_factor))

        # Generate the Repeated PRBS7Q Sync Header
        bits127 = prbs7()
        bits254 = bits127 + bits127  # Double the sequence to 254 bits

        # Map the 254 bits into 127 PAM4 symbols.
        # Using standard Gray Code mapping: 00->-3, 01->-1, 11->+1, 10->+3
        pam4_map = {(0, 0): -3.0, (0, 1): -1.0, (1, 1): 1.0, (1, 0): 3.0}
        prbs_symbols = [pam4_map[(bits254[2 * i], bits254[2 * i + 1])] for i in range(127)]
        prbs_symbols = np.array(prbs_symbols, dtype=np.float32)

        # "and repeat that" - Stack the 127-symbol sequence twice
        # This gives you 254 PAM4 symbols total for the correlation detector
        sync_block = np.concatenate((prbs_symbols, prbs_symbols))

        # Generate random PAM4 symbols
        remaining_symbols = num_symbols - len(sync_block)
        if remaining_symbols > 0:
            random_symbols = np.random.choice([-3.0, -1.0, 1.0, 3.0], size=remaining_symbols).astype(np.float32)
            symbols = np.concatenate((sync_block, random_symbols))
        else:
            symbols = sync_block[:num_symbols]

        # Upsample by zero insertion
        # tx_waveform = np.zeros(len(symbols) * self.os_factor, dtype=np.float32)
        # tx_waveform[::self.os_factor] = symbols

        # Upsample Zero order Hold (ZOH)
        tx_waveform = np.repeat(symbols, self.os_factor).astype(np.float32)

        # Apply smooth start ramp (Half-Hann window over 16 symbols)
        ramp_symbols = 16
        ramp_samples = ramp_symbols * self.os_factor
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(ramp_samples) / ramp_samples)).astype(np.float32)
        tx_waveform[:ramp_samples] *= ramp

        # Insert integer delay (in analog samples)
        if integer_delay_samples > 0:
            pad_zeros = np.zeros(integer_delay_samples, dtype=np.float32)
            tx_waveform = np.concatenate((pad_zeros, tx_waveform))

        # Pulse shape using batched FFT convolution
        # taps = generate_rrc_taps(self.os_factor, beta=0.35, span=8)

        # Apply physical hardware roll-off (4th-order Bessel-Thomson)
        # cutoff_ratio=0.75 models a typical 75 GHz Tx DAC bandwidth
        taps = generate_bessel_taps(self.os_factor, cutoff_ratio=2.0, span=16)

        tx_analog = fft_convolve_same(tx_waveform, taps)

        if False:
            from utils.plot_sig import plot_psd
            # Plot the PSD
            plot_psd(
                tx_waveform=tx_waveform,
                tx_filtered=tx_analog,
                sample_rate=self.baud_rate * self.os_factor,
                baud_rate=self.baud_rate,
                filter_cutoff_ghz=200
            )

        return symbols, tx_waveform, tx_analog, sync_block
