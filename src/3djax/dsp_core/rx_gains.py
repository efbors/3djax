import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


class RxGains:
    def __init__(self, config):
        self.ideal_levels = np.array([-3.0, -1.0, 1.0, 3.0])
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

#
# class RxGains:
#     def __init__(self, config):
#         self.ideal_levels = np.array([-3.0, -1.0, 1.0, 3.0])
#
# def calc_post_adc_gain(self, adc_samples, show_plot=False):
#     """
#             Calculates the optimal digital multiplier by comparing the true
#             baud-rate symbol RMS against the ideal theoretical PAM-4 target.
#             """
#
#
#
#     counts, bin_edges = np.histogram(adc_samples, bins=200)
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#
#     # Find Peaks
#     peaks, properties = find_peaks(counts, prominence=np.max(counts) * 0.1, distance=10)
#
#     if len(peaks) < 4:
#         print("WARNING: Found fewer than 4 peaks. Returning fallback gain.")
#         rms = np.sqrt(np.mean(adc_samples ** 2))
#         return np.float32(2.236 / (rms + 1e-6))
#
#     # Get the 4 highest peaks (in case of noise spikes) and sort left to right
#     top_4_idx = peaks[np.argsort(counts[peaks])[-4:]]
#     top_4_idx = np.sort(top_4_idx)
#
#     actual_peaks = bin_centers[top_4_idx]
#     confidences = counts[top_4_idx]  # Bin height is our confidence weight!
#
#     # Weighted Least Squares Calculation for Optimal Gain
#     # Goal: minimize the error: Error = sum( w * (ideal - gain * actual_peak)^2 )
#     # Taking the derivative with respect to gain and setting to 0 yields:
#     # gain = sum( w * actual_peaks * ideal_levels ) / sum( w * ideal_levels^2 )
#
#     numerator = np.sum(confidences * actual_peaks * self.ideal_levels)
#     denominator = np.sum(confidences * (self.ideal_levels ** 2))  # FIX: Base the denominator on ideal_levels
#     optimal_gain = numerator / denominator
#
#     return np.float32(optimal_gain)
