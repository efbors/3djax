import numpy as np
from scipy.interpolate import CubicSpline

class APhaseInterp:
    def __init__(self, config):
        self.config = config
        self.pi_resolution = int(self.config['rx']['pi_resolution'])
        self.pi_eval_block_size = int(self.config['rx']['pi_eval_block_size'])
        self.baud_rate = float(self.config['system']['baud_rate'])
        self.os_factor = float(self.config['system']['os_factor'])

    def calc_coarse_phase(self, base_sample_times, rx_afe_out):

        """
        Calculates the optimal fractional phase offset by sweeping the unit interval,
        interpolating the analog waveform, and finding the phase with maximum variance.

        Args:
            base_sample_times (np.array): The full 1D continuous analog waveform from the AFE.
            rx_afe_out (np.array): 1D array of baseline fractional sampling indices for the eval block.

        Returns:
            np.float32: The fractional analog index offset that centers the eye.
        """
        # Create the continuous-time spline representation of the AFE output
        x_analog = np.arange(len(rx_afe_out))
        cs = CubicSpline(x_analog, rx_afe_out)

        # Generate the discrete phase offset steps covering 1 Unit Interval (UI)
        # 1 UI spans exactly 'self.os_factor' analog indices
        phase_offsets = np.linspace(0.0, self.os_factor, self.pi_resolution, endpoint=False)

        # Vectorized Grid Generation (The "No Loop" Magic)
        # base_sample_times[:, np.newaxis] turns an (N,) array into an (N, 1) matrix.
        # Adding the (R,) offsets broadcasts it into a massive (N, R) grid of fractional times.
        eval_grid = base_sample_times[:, np.newaxis] + phase_offsets

        # Safely clip to prevent the Spline from throwing boundary extrapolation errors
        eval_grid = np.clip(eval_grid, 0.0, len(rx_afe_out) - 1.0)

        # One-Shot Interpolation
        # Flatten the (N, R) grid, run the C-optimized Spline once, and reshape back to (N, R)
        eval_grid_flat = eval_grid.ravel()
        interpolated_flat = cs(eval_grid_flat)
        sampled_matrix = interpolated_flat.reshape(-1, self.pi_resolution)

        # Variance Metric Calculation
        # Calculate the variance down the columns (axis=0). Results in an array of shape (R,)
        variances = np.var(sampled_matrix, axis=0)

        # Find the peak variance and return its corresponding fractional offset
        best_idx = np.argmax(variances)
        best_offset = np.float32(phase_offsets[best_idx])

        return variances, best_offset
