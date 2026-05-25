import numpy as np

class RxGains:
    def __init__(self, config):
        self.ideal_levels = config['system']['ideal_levels']
        # Pull the oversampling factor directly from the configuration
        self.os_factor = int(config['system']['os_factor'])

    def calc_post_adc_gain(self, adc_samples, show_plot=False):
        """
        Calculates the optimal digital multiplier by comparing the true
        baud-rate symbol RMS against the ideal theoretical PAM-4 target.
        """
        # 1. Extract ONLY the eye centers (1 sample per symbol)
        centered_symbols = adc_samples[0::self.os_factor]

        # 2. Calculate the measured RMS of the true symbols
        measured_rms = np.sqrt(np.mean(centered_symbols ** 2))

        # 3. Target RMS for a perfect PAM-4 alphabet (sqrt(5))
        target_rms = np.sqrt(5.)

        # 4. Calculate the exact gain multiplier needed
        optimal_gain = target_rms / (measured_rms + 1e-6)

        return np.float32(optimal_gain)


