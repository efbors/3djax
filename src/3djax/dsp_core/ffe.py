""" FFE: block feed forward equalizer
"""
import numpy as np


class FFE:
    def __init__(self, config):
        ffe_cfg = config['rx']['FFE']

        # Architecture Parameters
        self.num_taps = int(ffe_cfg.get('ffe_taps'))
        self.mu = float(ffe_cfg.get('mu'))
        self.ideal_levels = config['system']['ideal_levels']

        # State Memory (The overlap buffer)
        # To filter a block continuously, we must remember the last 'num_taps - 1'
        # symbols from the previous block.
        self.history = np.zeros(self.num_taps - 1, dtype=np.float32)

        # Precursor percent determines the approximate target of the
        # taps center of mass (COM);  please see the docs for more details
        self.precursor_percent = ffe_cfg['precursor_percent']
        # Calculate the main cursor index based on precursor percentage
        # e.g., 15% of 20 (for 21 taps) = index 3
        float_idx = (self.num_taps - 1) * (self.precursor_percent / 100.0)
        self.center_idx = int(round(float_idx))
        pre_taps = self.center_idx
        post_taps = self.num_taps - 1 - self.center_idx

        self.coeff = np.zeros(self.num_taps, dtype=np.float32)
        self.coeff[self.center_idx] = 1.0

        # --- regularization setup (Edge Repulsion)
        # This block attempts to keep the center of mass of the FFE coefficients
        # away from the edges;
        # Apply the 0.8 heuristic
        W = int(round(0.8 * min(pre_taps, post_taps)))
        # Force W to be even for perfect symmetry
        if W % 2 != 0:
            W += 1
        half_W = W // 2
        # Generate the full window (e.g., beta=5.0 for a smooth but firm curve)
        kaiser_beta = 5.0
        full_kaiser = np.kaiser(W, kaiser_beta)

        # Invert and scale to create the penalty shape
        self.alpha = float(ffe_cfg['alpha'])
        penalty_shape = self.alpha * (1.0 - full_kaiser)

        # Split into left and right walls
        left_wall = penalty_shape[:half_W]
        right_wall = penalty_shape[half_W:]

        # Apply the left/right penalty walls
        self.penalty = np.zeros(self.num_taps, dtype=np.float32)
        self.penalty[:half_W] = left_wall
        self.penalty[-half_W:] = right_wall

    def calc(self, ffe_in):
        """
        Executes the Block LMS forward pass and weight update,
        and metric extraction.
        """
        # concatenate history with inputs
        ffe_inx = np.concatenate((self.history, ffe_in))

        # Update history block
        self.history = ffe_in[-(self.num_taps - 1):]

        # mode='valid' so the output length matches ffe_in length N
        ffe_out = np.convolve(ffe_inx, self.coeff, mode='valid')

        # Slice ffe_out to the reference levels via broadcasted distance calculation
        distances = np.abs(ffe_out[:, np.newaxis] - self.ideal_levels)
        ffe_slice = self.ideal_levels[np.argmin(distances, axis=1)]

        ffe_error = ffe_slice - ffe_out

        # Compute the block LMS gradient using cross-correlation.
        # np.correlate(..., mode='valid') executes the exact sum(e * x) for every tap offset
        # without needing multi-dimensional arrays
        gradient = np.correlate(ffe_inx, ffe_error, mode='valid')

        # Update coefficients: Apply edge-repelling penalty, then add the gradient
        self.coeff = self.coeff * (1.0 - self.penalty) + (self.mu * gradient)

        # -- Metric Extractions
        # Calculate Center of Mass (COM)
        power = self.coeff ** 2
        com = np.sum(np.arange(self.num_taps) * power) / np.sum(power)
        com_error = self.center_idx - com

        # Calculate Newton-Based Phase Error using the three center taps
        w1 = self.coeff[self.center_idx - 1]
        w2 = self.coeff[self.center_idx]
        w3 = self.coeff[self.center_idx + 1]

        denominator = 2.0 * (w1 - 2.0 * w2 + w3)

        # Guard against zero-division if taps are perfectly flat
        if abs(denominator) > 1e-9:
            phase_error = (w1 - w3) / denominator
        else:
            phase_error = 0.0

        return ffe_out, phase_error, com_error
