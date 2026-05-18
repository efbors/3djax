import numpy as np

# class FrequencySynthesizer:
#     def __init__(self, config, nsamp_os):
#         self.config = config
#         self.nsamp = nsamp_os
#
#     def gen_sample_times(self, nsamp):
#         # Calculate the fractional, sampling times at symbol rate
#         self.baud_rate = int(self.config['system']['baud_rate'])
#         self.os_factor = int(self.config['system']['os_factor'])
#         self.frequency_offset_ppm = int(self.config['rx']['frequency_offset_ppm'])
#         self.jitter_rms_ui = int(self.config['rx']['jitter_rms_ui'])
#



class FrequencySynthesizer:
    def __init__(self, config, nsamp_os):
        self.config = config
        self.nsamp = nsamp_os
        self.os_factor = int(self.config['system']['os_factor'])
        self.baud_rate = float(self.config['system']['baud_rate'])

        # Cast to float, not int, to preserve the 57.9 and 0.02 values
        self.frequency_offset_ppm = float(self.config['rx']['frequency_offset_ppm'])
        self.jitter_rms_ui = float(self.config['rx']['jitter_rms_ui'])

    def gen_sample_times(self, nsamp):
        """
        Calculates the fractional analog array indices where the ADC should sample,
        incorporating the receiver's local clock drift (PPM) and phase noise (Jitter).
        """

        eps = self.frequency_offset_ppm * 1e-6

        # Calculate the Receiver's Step Size
        # If the Rx clock is faster (+PPM), its symbol period is shorter.
        # In terms of analog array indices, the distance between samples is slightly less than os_factor.
        rx_step_size = self.os_factor / (1.0 + eps)

        # Define the Base Clock Grid
        # Calculate roughly how many full symbols exist in the analog waveform
        num_symbols = int(nsamp // rx_step_size) - 1

        # The nominal, drifting clock ticks (k = 0, 1, 2...)
        k = np.arange(num_symbols, dtype=np.float64)
        base_indices = k * rx_step_size

        # Inject Physical Jitter (Phase Noise)
        # jitter_rms_ui is a fraction of a symbol (UI).
        # We multiply by os_factor to convert that UI fraction into analog array indices.
        if self.jitter_rms_ui > 0.0:
            jitter_indices = np.random.normal(0.0, self.jitter_rms_ui * self.os_factor, size=num_symbols)
        else:
            jitter_indices = 0.0

        sample_indices = base_indices + jitter_indices

        # Clean up boundaries
        # The Cubic Spline will crash if we ask it to interpolate an index outside the bounds of the array
        valid_mask = (sample_indices >= 0.0) & (sample_indices <= (nsamp - 1))
        valid_sample_indices = sample_indices[valid_mask]

        # Return as float32 to save memory in the DSP blocks
        return valid_sample_indices.astype(np.float32)