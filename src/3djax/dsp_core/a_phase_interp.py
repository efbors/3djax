import numpy as np
from scipy.interpolate import CubicSpline


class APhaseInterp:
    def __init__(self, config):
        self.config = config
        self.pi_resolution = int(self.config['rx']['api']['pi_resolution'])
        self.pi_eval_block_size = int(self.config['rx']['api']['pi_eval_block_size'])
        self.baud_rate = float(self.config['system']['baud_rate'])
        self.os_factor = float(self.config['system']['os_factor'])

    def calc_coarse_phase(self, api_in):
        """
        Calculates the optimal fractional phase offset by sweeping the unit interval,
        interpolating the analog waveform, and finding the phase with maximum variance.

        Returns:
            tuple: (variances array, best fractional analog index offset)
        """
        # Create the continuous-time spline representation of the input block
        x_analog = np.arange(len(api_in))
        cs = CubicSpline(x_analog, api_in)

        # Generate nominal symbol sampling times, strictly spaced by os_factor
        base_sample_times = np.arange(0, len(api_in), self.os_factor)

        # Generate the discrete phase offset steps covering 1 Unit Interval (UI)
        # 1 UI spans exactly 'self.os_factor' analog indices
        phase_offsets = np.linspace(0.0, self.os_factor, self.pi_resolution, endpoint=False)

        # Vectorized Grid Generation (The "No Loop" Magic)
        # base_sample_times[:, np.newaxis] turns an (N,) array into an (N, 1) matrix.
        # Adding the (R,) offsets broadcasts it into a massive (N, R) grid of fractional times.
        eval_grid = base_sample_times[:, np.newaxis] + phase_offsets

        # Safely clip to prevent the Spline from throwing boundary extrapolation errors
        eval_grid = np.clip(eval_grid, 0.0, len(api_in) - 1.0)

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

    def sample_block(self, rx_in, index, timing_offset, nsamp, stride):
        """
        Extracts a block of interpolated samples by evaluating the
        continuous-time spline representation of the input signal at the
        desired fractional sample phase.
        Note: uses one stride to the left, one to the right of the extracted block.

        :param rx_in:  input time domain series (1D numpy array)
        :param index:  starting base integer index in the input array
        :param timing_offset: fractional offset (float) to shift the sampling phase
        :param nsamp: number of interpolated samples to generate
        :param stride: stride (analog index distance) between the samples (e.g., os_factor)
        :return: 1D numpy array of length 'len' containing the interpolated samples
        """
        fractional_offset = timing_offset - int(np.floor(timing_offset))

        # Define the local block bounds with a 1-stride buffer on both sides
        rx_start = index - stride
        rx_len = (nsamp + 2) * stride

        # Extract the raw analog sample chunk
        rx_blk_in = rx_in[rx_start: rx_start + rx_len]

        # Map the local knot coordinates (simple integer spacing from 0 to rx_len)
        x_knots = np.arange(len(rx_blk_in))

        # Fit the cubic spline to this localized chunk
        cs = CubicSpline(x_knots, rx_blk_in)

        # Generate the precise local evaluation times
        # Relative to rx_start (0), the base 'index' sits exactly at position 'stride'.
        # We start at 'stride + fractional_offset' and step by 'stride' for 'nsamp' items.
        eval_times = stride + fractional_offset + (np.arange(nsamp) * stride)

        # 4. One-shot evaluation directly at our target points
        interpolated_samples = cs(eval_times)

        return interpolated_samples.astype(np.float32)

