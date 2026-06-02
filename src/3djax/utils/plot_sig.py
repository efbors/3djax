import matplotlib.pyplot as plt
import numpy as np


class DJPlot:
    def __init__(self, title="", xlabel=""):
        self.signals = []
        self.vlines = []
        self.colors = ['r', 'g', 'b', 'k', 'y']
        self.color_idx = 0
        self.max_len = 0
        self.title = title
        self.xlabel = xlabel

    def add_plot(self, vector, label_string, is_step=False, **kwargs):
        """Adds a signal to the plot queue. vector acts as the Y-axis data."""
        vector = np.asarray(vector)
        self.max_len = max(self.max_len, len(vector))

        # Cycle through 'r', 'g', 'b', 'k', 'y' if color isn't explicitly passed
        if 'color' not in kwargs:
            kwargs['color'] = self.colors[self.color_idx % len(self.colors)]
            self.color_idx += 1

        self.signals.append({
            'y': vector,
            'label': label_string,
            'is_step': is_step,
            'kwargs': kwargs
        })

    def add_vlines(self, x_coords, ymin, ymax, label_string, **kwargs):
        """Queues vertical lines."""
        self.vlines.append({
            'x': np.asarray(x_coords),
            'ymin': ymin,
            'ymax': ymax,
            'label': label_string,
            'kwargs': kwargs
        })

    def do_plot(self):
        """Renders all queued plots onto a single figure."""
        plt.figure(figsize=(18, 8))

        # Plot signals
        for sig in self.signals:
            y = sig['y']
            # Implicit X-axis basis up to the length of the vector
            x = np.arange(len(y))

            if sig['is_step']:
                plt.step(x, y, label=sig['label'], **sig['kwargs'])
            else:
                plt.plot(x, y, label=sig['label'], **sig['kwargs'])

        # Plot vertical lines
        for vl in self.vlines:
            plt.vlines(vl['x'], ymin=vl['ymin'], ymax=vl['ymax'], label=vl['label'], **vl['kwargs'])

        if self.title:
            plt.title(self.title)
        if self.xlabel:
            plt.xlabel(self.xlabel)

        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()


def plot_acq_times(start_ix, win_len,
                   wave_0, label_0,
                   take_times,
                   title=None):
    plt.figure(figsize=(18, 8))

    # Define the window
    end_ix = start_ix + win_len
    x_vals = np.arange(start_ix, end_ix)
    wave_slice = wave_0[start_ix:end_ix]

    # Plot the main analog waveform
    plt.plot(x_vals, wave_slice, label=label_0, color='blue')

    # Filter take_times to only include those within the current plot window
    take_times = np.asarray(take_times)
    valid_take_times = take_times[(take_times >= start_ix) & (take_times < end_ix)]

    # Add the vertical lines
    if len(valid_take_times) > 0:
        ymin, ymax = plt.ylim()  # Match the height of the waveform
        plt.vlines(valid_take_times, ymin=ymin, ymax=ymax,
                   colors='red', linestyles='dashed', alpha=0.7,
                   label='Sample Times')

    # Formatting
    if title:
        plt.title(title)
    plt.xlabel('Analog Indices')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show(block=False)


def plot_time_domain(start_ix, win_len,
                     wave_0, label_0=None,
                     wave_1=None, label_1=None,
                     wave_2=None, label_2=None,
                     title=None
                     ):
    plt.figure(figsize=(18, 8))
    lbl_0 = label_0 if label_0 else 'wave_0'
    plt.plot(wave_0[start_ix:start_ix + win_len], 'r', label=lbl_0)
    if wave_1 is not None:
        lbl_1 = label_1 if label_1 else 'wave_1'
        plt.plot(wave_1[start_ix:start_ix + win_len], 'g', label=lbl_1)
    if wave_2 is not None:
        lbl_2 = label_2 if label_2 else 'wave_2'
        plt.plot(wave_2[start_ix:start_ix + win_len], 'b', label=lbl_2)
    title = title if title else 'Time Domain Plot'
    plt.title(title)
    plt.xlabel("Analog Indices")
    plt.ylabel("V")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show(block=False)


def plot_sig():

    # Ensure start is an integer for valid array slicing
    os_factor = int(config['system']['os_factor'])
    start_offset = 15000
    start = int(start_sym * os_factor) + start_offset

    winlen = 3000

    plt.figure(figsize=(18, 8))
    plt.plot(tx_analog[start:start + winlen], 'r', label='Tx Analog')
    plt.plot(channel_out[start:start + winlen], 'g', label='Channel Out')
    plt.plot(rx_afe_out[start:start + winlen], 'b', label='Rx AFE Out (CTLE)')
    plt.plot(vga_gain[start:start + winlen], 'k', label='VGA Gain')
    plt.plot(sd_flag[start:start + winlen], 'y', label='Signal Detect')

    # Overlay the state machine                               # <--- INSERTED STATE MACHINE PLOT
    plt.step(np.arange(winlen), sm_state[start:start + winlen], color='orange', linewidth=3, label='State Machine')

    # Filter for only the clock ticks that land inside our 3000-index window
    samples_in_window = rx_sample_times[(rx_sample_times >= start) & (rx_sample_times < start + winlen)]
    plot_sample_x = samples_in_window - start

    # Draw dashed vertical lines at the exact moments the PI decided to sample
    plt.vlines(plot_sample_x, ymin=-3.5, ymax=3.5, color='magenta',
               linestyle='--', alpha=0.7,
               label='Sample Moments')

    plt.title("Time Domain: AFE Out with Optimal PI Sampling Phases")
    plt.xlabel(f"Analog Indices (Relative to absolute index {start})")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


def plot_psd(tx_waveform, tx_filtered, sample_rate, baud_rate, filter_cutoff_ghz):
    import matplotlib.pyplot as plt
    from scipy.signal import bessel, lfilter, welch

    # Calculate Power Spectral Density
    # nperseg=8192 : balance of frequency resolution and noise averaging
    f_zoh, psd_zoh = welch(tx_waveform, fs=sample_rate, nperseg=8192)
    f_filt, psd_filt = welch(tx_filtered, fs=sample_rate, nperseg=8192)

    # Convert to dB (add 1e-15 floor to prevent log10(0))
    psd_zoh_db = 10 * np.log10(np.maximum(psd_zoh, 1e-15))
    psd_filt_db = 10 * np.log10(np.maximum(psd_filt, 1e-15))

    # Plot the Spectrum
    plt.figure(figsize=(14, 7))

    # Plot the raw ZOH upsampled signal
    plt.plot(f_zoh / 1e9, psd_zoh_db, label='ZOH Only (Unfiltered)', alpha=0.6, color='gray')

    # Plot the Bessel-filtered signal
    plt.plot(f_filt / 1e9, psd_filt_db, label='ZOH + 200 GHz Bessel Filter', color='red')

    # Formatting
    plt.title('Transmitter Output Spectrum: 100 Gbaud PAM4 sampled at 1.6 THz')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Power Spectral Density (dB/Hz)')
    plt.grid(True, which="both", ls="--", alpha=0.5)

    # Limit X-axis to Nyquist of the sample rate
    plt.xlim(0, 800)
    plt.ylim(-180, np.max(psd_zoh_db) + 5)

    # Annotations
    plt.axvline(50, color='blue', linestyle='--', alpha=0.8, label='Nyquist Fundamental (50 GHz)')
    plt.axvline(100, color='green', linestyle='-.', alpha=0.8, label='Baud Rate / ZOH Notches')

    # Draw remaining ZOH notches
    for f in [200, 300, 400, 500, 600, 700]:
        plt.axvline(f, color='green', linestyle='-.', alpha=0.3)

    plt.axvline(200, color='purple', linestyle=':', linewidth=2, label='Filter -3dB Cutoff (200 GHz)')

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()



def plot_channel_batch(h_tensor, dt):
    """
    Overlays all channel impulse responses in the batch to visualize the
    statistical distortion envelope created by your YAML min/max configurations.

    :param h_tensor: 2D numpy array of shape (B, max_batch_len) from the channel object
    :param dt: Sample time resolution in seconds (channel.dt)
    """
    B, max_len = h_tensor.shape

    # Convert time axis to picoseconds (1 UI @ 100GBd = 10 ps)
    t_ps = np.arange(max_len) * dt * 1e12

    plt.figure(figsize=(11, 5))

    # Plot each batch realization
    for b in range(B):
        # Using alpha transparency helps visualize overlapping high-density regions
        plt.plot(t_ps, h_tensor[b], alpha=0.5, linewidth=1.5,
                 label=f'Realization {b}' if B <= 8 else None)

    plt.title(f'RefDisRes Channel Impulse Responses ($h$) — Batch Size: {B}', fontsize=12, fontweight='bold')
    plt.xlabel('Time (ps)', fontsize=11)
    plt.ylabel('Normalized Amplitude (V)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)

    # Clean up layout depending on how many lines are drawn
    if B <= 8:
        plt.legend(loc='upper right', fontsize=9)
    else:
        # If batch is huge, don't choke the plot area with a massive legend
        plt.text(0.02, 0.03, f'* Showing all {B} unique randomized channel topologies',
                 transform=plt.gca().transAxes, fontsize=9, style='italic',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    plt.show(block=False)
