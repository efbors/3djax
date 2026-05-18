import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import lfilter


class AnalogFrontEnd:
    def __init__(self, config):
        self.config = config
        self.baud_rate = float(config['system']['baud_rate'])
        self.os_factor = int(config['system']['os_factor'])
        self.fs = self.baud_rate * self.os_factor
        self.dt = 1 / self.fs

        # AFE Knobs (Normally these would be in your YAML under 'afe')
        self.target_amplitude = 1.0  # Target peak-to-peak amplitude for the ADC
        self.squelch_threshold = 0.05  # 50mV threshold to declare "Signal Present"
        self.agc_time_constant_ui = 1000  # very cold AGC. Takes ~1000 UI to adjust

        # CTLE Knobs
        self.ctle_dc_gain = 0.3
        self.ctle_f_z = 8e9
        self.ctle_f_p = 37e9
        self.ctle_f_p2 = 800e9  # remove it for now

    def process(self, rx_pad_analog):
        """
        Passes the raw pad voltage through the Signal Detect, VGA, and CTLE.
        """
        # ---------------------------------------------------------
        # 1. ENVELOPE DETECTION & SIGNAL DETECT (SQUELCH)
        # ---------------------------------------------------------
        # 1-Pole IIR Filter (RC Low-pass)
        # alpha = dt / tau. A smaller alpha means a slower/heavier filter.
        envelope_alpha = 1.0 / (self.os_factor * 50)

        # lfilter coefficients: b = [alpha], a = [1.0, -(1.0 - alpha)]
        rx_envelope = lfilter([envelope_alpha], [1.0, -(1.0 - envelope_alpha)], np.abs(rx_pad_analog))

        # Signal Detect Flag: 0.75 if active, 0.0 elsewhere
        signal_present = np.where(rx_envelope > self.squelch_threshold, 0.75, 0.0)

        # ---------------------------------------------------------
        # 2. THE VGA (Variable Gain Amplifier) / AGC LOOP
        # ---------------------------------------------------------
        safe_envelope = np.clip(rx_envelope, 1e-6, None)
        ideal_gain = self.target_amplitude / safe_envelope

        # AGC is very slow. Very small alpha.
        agc_alpha = 1.0 / (self.os_factor * self.agc_time_constant_ui)

        actual_gain = lfilter([agc_alpha], [1.0, -(1.0 - agc_alpha)], ideal_gain)

        actual_gain = np.where(signal_present, actual_gain, 1.0)
        rx_vga = rx_pad_analog * actual_gain

        # ---------------------------------------------------------
        # 3. THE CTLE (Continuous Time Linear Equalizer)
        # ---------------------------------------------------------
        f = np.fft.fftfreq(len(rx_vga), d=self.dt)
        s = 2j * np.pi * f
        w_z = 2 * np.pi * self.ctle_f_z
        w_p = 2 * np.pi * self.ctle_f_p
        w_p2 = 2 * np.pi * self.ctle_f_p2
        # Notice the second pole is (1 + s / w_p2)
        H_ctle = self.ctle_dc_gain * (s + w_z) / (s + w_p) / (1.0 + s / w_p2)

        # The DC gain calculation goes back to the standard 1-pole math
        # because the 2nd pole evaluates to exactly 1.0 at DC!
        H_ctle[0] = self.ctle_dc_gain * (self.ctle_f_z / self.ctle_f_p)

        rx_fft = np.fft.fft(rx_vga)
        rx_ctle = np.real(np.fft.ifft(rx_fft * H_ctle))

        return rx_ctle, signal_present, actual_gain
