import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


class RxGains:
    def __init__(self, config):
        self.ideal_levels = np.array([-3.0, -1.0, 1.0, 3.0])

    def calc_post_adc_gain(self, adc_samples, show_plot=False):
        # 1. Generate Histogram
        counts, bin_edges = np.histogram(adc_samples, bins=200)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 2. Find Peaks
        peaks, properties = find_peaks(counts, prominence=np.max(counts) * 0.1, distance=10)

        if len(peaks) < 4:
            print("WARNING: Found fewer than 4 peaks. Returning fallback gain.")
            rms = np.sqrt(np.mean(adc_samples ** 2))
            return np.float32(2.236 / (rms + 1e-6))

        # Get the 4 highest peaks (in case of noise spikes) and sort left to right
        top_4_idx = peaks[np.argsort(counts[peaks])[-4:]]
        top_4_idx = np.sort(top_4_idx)

        actual_peaks = bin_centers[top_4_idx]
        confidences = counts[top_4_idx]  # Bin height is our confidence weight!

        # 3. Weighted Least Squares Calculation for Optimal Gain
        numerator = np.sum(confidences * actual_peaks * self.ideal_levels)
        denominator = np.sum(confidences * (actual_peaks ** 2))
        optimal_gain = numerator / denominator

        if show_plot:
            plt.figure(figsize=(10, 5))
            plt.bar(bin_centers, counts, width=(bin_edges[1] - bin_edges[0]), color='gray', alpha=0.7)
            plt.plot(bin_centers[top_4_idx], counts[top_4_idx], "rx", markersize=10, label='Detected Peaks')
            for i in range(4):
                plt.text(actual_peaks[i], counts[top_4_idx[i]] * 1.05,
                         f"Volts: {actual_peaks[i]:.2f}\nConf: {confidences[i]}",
                         ha='center', color='red', weight='bold')
            plt.title(f"ADC Output Histogram\nCalculated Weighted Digital Gain: {optimal_gain:.3f}")
            plt.xlabel("ADC Voltage")
            plt.ylabel("Occurrences (Confidence)")
            plt.grid(True)
            plt.legend()
            plt.show()

        return np.float32(optimal_gain)