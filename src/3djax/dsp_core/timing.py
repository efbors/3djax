""" timing.py
implement a proportional-integral second order loop to calculate timing
(please see docs for the block diagram and more information).

"""
import numpy as np


class Timing:
    def __init__(self, config):
        # Extract configurations (using defaults if not yet defined in YAML)
        timing_cfg = config.get('timing', {})

        # Loop Gains
        self.alpha = float(timing_cfg.get('alpha', 1e-3))  # Fast Proportional
        self.beta = float(timing_cfg.get('beta', 1e-5))  # Frequency Integral
        self.gamma = float(timing_cfg.get('gamma', 1e-6))  # Cold Steering Integral

        # Determine the ideal Center of Mass (COM) target.
        # If your FFE is already passing out the fractional 'com_error'
        # instead of the raw COM, you can just set this to 0.0.
        ffe_cfg = config.get('FFE', {})
        num_taps = int(ffe_cfg.get('ffe_taps', 21))
        precursor_percent = float(ffe_cfg.get('precursor_percent', 40.0))
        self.ideal_com = round((num_taps - 1) * (precursor_percent / 100.0))

        # Independent Integrator States
        self.I1_state = 0.0  # Tracks PPM / Master Phase Offset
        self.I2_state = 0.0  # Tracks Cold Steering / Channel Delay Bias

        # State to track the integer slip applied in the previous block
        self.applied_slip = 0

    def calc(self, phase_error, com):
        """
        Calculates the per-block timing offset using a split-integrator loop.

        :param phase_error: Instantaneous phase error (e.g., from Newton estimation)
        :param com: Actual Center of Mass of the FFE taps (or com_error)
        :return: timing_offset; fractional [-1.5,+1.5]; fractional part represents the
        phase error. integer part {-1,0,+1} represents the slip.
        """
        # 1. Feedback: Deduct the slip applied in the previous block.
        # This prevents the master accumulator from winding up to infinity.
        self.I1_state -= self.applied_slip

        # 2. The Fast Path (Proportional)
        prop_out = self.alpha * phase_error

        # 3. The Frequency Path (Integrator 1)
        self.I1_state += (self.beta * phase_error)

        # Anti-windup safeguard for the master phase accumulator
        self.I1_state = np.clip(self.I1_state, -10.0, 10.0)

        # 4. The Steering Path (Integrator 2)
        # Calculate displacement error (Crucial fix: do not integrate raw COM!)
        com_error = com - self.ideal_com
        self.I2_state += (self.gamma * com_error)

        # Anti-windup safeguard for the steering accumulator
        self.I2_state = np.clip(self.I2_state, -2.0, 2.0)

        # 5. Combine for Master Output
        timing_offset = prop_out + self.I1_state + self.I2_state

        # 6. Extract the integer slip to feed back on the next block.
        # np.round() natively handles the standard -0.5 to +0.5 UI thresholding.
        self.applied_slip = int(np.round(timing_offset))

        return timing_offset
