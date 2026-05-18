import numpy as np


class ADC:
    def __init__(self, config):
        rx_cfg = config['rx']
        # The true signal-to-noise quality metric
        self.enob = float(rx_cfg.get('ADC_enob', rx_cfg.get('adc_quantization_enob', 5.6)))

        # The hard clipping limits (+/- Volts)
        self.clip_limit = float(rx_cfg.get('ADC_input_max', 8.0))

        # The actual physical hardware datapath (Defaulting to 8 bits if not in yaml)
        self.physical_bits = int(rx_cfg.get('adc_physical_bits', 8))
        self.num_levels = int(2 ** self.physical_bits)  # Exactly 256 levels

        # --- ENOB Noise Calculation (IEEE Standard) ---
        # Calculate required SNR in dB
        snr_db = 6.02 * self.enob + 1.76
        snr_linear = 10 ** (snr_db / 10.0)

        # Assuming full-scale sine wave for standard ENOB definition
        signal_power = (self.clip_limit ** 2) / 2.0

        # Calculate internal ADC noise standard deviation
        self.noise_variance = signal_power / snr_linear
        self.noise_std = np.sqrt(self.noise_variance)

    def process(self, analog_samples, pre_gain=1.0):
        """
        Simulates physical ADC by injecting ENOB-equivalent noise, 
        clipping, and binning to the physical bit depth.
        """
        # Apply analog pre-gain (VGA target)
        scaled = analog_samples * pre_gain

        # Inject ENOB-equivalent internal noise BEFORE quantization
        noisy_signal = scaled + np.random.normal(0, self.noise_std, size=scaled.shape)

        # Hard Clip (Saturation)
        clipped = np.clip(noisy_signal, -self.clip_limit, self.clip_limit)

        # Physical Quantization to exact 2^N bins
        norm = (clipped + self.clip_limit) / (2 * self.clip_limit)
        binned = np.round(norm * (self.num_levels - 1))
        quantized_volts = (binned / (self.num_levels - 1)) * (2 * self.clip_limit) - self.clip_limit

        return quantized_volts.astype(np.float32)
