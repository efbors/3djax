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
    # --- INSERTED BLOCK END ---
    if True:
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
