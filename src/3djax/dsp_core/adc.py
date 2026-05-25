import numpy as np


class ADC:
    def __init__(self, config):
        adc_cfg = config['rx']['ADC']

        # The hard clipping limits (+/- Volts)
        self.clip_limit = adc_cfg['clip_limit']
        self.physical_bits = adc_cfg['physical_bits']
        self.enob = min(adc_cfg['enob'], float(self.physical_bits))
        self.num_levels = int(2 ** self.physical_bits)

        # ENOB Noise Calculation (IEEE Standard)
        # Calculate required SNR in dB
        snr_db = 6.02 * self.enob + 1.76
        snr_linear = 10 ** (snr_db / 10.0)

        # Assuming full-scale sine wave for standard ENOB definition
        signal_power = (self.clip_limit ** 2) / 2.0

        # Calculate internal ADC noise standard deviation
        self.noise_variance = signal_power / snr_linear
        self.noise_std = np.sqrt(self.noise_variance)

    def process(self, analog_samples):
        """
        Simulates physical ADC by injecting ENOB-equivalent noise, 
        clipping, and binning to the physical bit depth.
        """

        # Inject ENOB-equivalent internal noise before quantization
        noisy_signal = analog_samples + np.random.normal(0, self.noise_std, size=analog_samples.shape)

        # Hard Clip (Saturation)
        clipped = np.clip(noisy_signal, -self.clip_limit, self.clip_limit)

        # Physical Quantization to exact 2^N bins
        norm = (clipped + self.clip_limit) / (2 * self.clip_limit)
        binned = np.round(norm * (self.num_levels - 1))
        quantized_volts = (binned / (self.num_levels - 1)) * (2 * self.clip_limit) - self.clip_limit

        return quantized_volts.astype(np.float32)
