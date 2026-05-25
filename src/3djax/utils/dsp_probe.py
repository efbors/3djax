# utils/dsp_probe.py
import numpy as np
from numpy.lib.stride_tricks import as_strided
import multiprocessing
import time


class DspProbe:
    def __init__(self, config):
        self.config = config

        self.tools_cfg = config.get('tools', {})
        self.enable_gui = self.tools_cfg.get('enable_gui', False)

        # Internal staging area for accumulating data before a flush
        self._staging_buffer = {}
        self._last_flush_time = 0.0

        if self.enable_gui:
            queue_size = self.tools_cfg.get('queue_max_size', 5)
            self.queue = multiprocessing.Queue(maxsize=queue_size)

            # Import GUI here to avoid circular imports if they live in different files
            from utils.dsp_gui import DspGui

            self.gui_process = multiprocessing.Process(
                target=DspGui.launch,
                args=(self.config, self.queue)
            )
            self.gui_process.start()
        else:
            self.queue = None
            self.gui_process = None
            # TODO: Initialize your HDF5 / Cloud / Datalake logger here
            print("DspProbe: GUI disabled. Running in headless data-logging mode.")

    def clear(self, max_frames, fps, loop=False):
        """
        Starts a new probing session and configures the viewer timeline.
        """
        self.max_frames = max_frames
        self.fps = fps
        self.loop = loop

        # Reset counters and internal state
        self.frame_index = 0
        self._staging_buffer.clear()

        # Reset the pacing clock
        self._last_flush_time = time.time()

    def plot(self, name, frame_index, input_signal, gain,
             start_index, nrow, seg_size, ext_samples):
        """  add a frame to the plot window

        :param name:
        :param input_signal:
        :param gain:
        :param start_index:
        :param nrow:
        :param seg_size:
        :param row_extend:
        :return:
        """

        # Build the 2D frame;  each row is a path over time of points equally distant
        frame_2d = self._signal_frame(input_signal, start_index, nrow,
                                      seg_size, ext_samples)
        # Ensure the 'plot' category exists in the staging buffer
        if 'plot' not in self._staging_buffer:
            self._staging_buffer['plot'] = {}

        # Stage the data and its metadata using the specific signal name
        self._staging_buffer['plot'][name] = {
            'frame_2d': frame_2d,
            'frame_index': frame_index,
            'gain': gain
        }
        print(f"")

    def trace(self, vals, ymin, ymax):
        pass

    def spec(self):
        pass

    def scalar(self, index, name, value):
        pass

    def flush(self, frame_index):
        """
        Packages the staged data, enforces the simulation frame rate,
        dispatches the payload, and resets for the next frame.
        """
        # Dynamic Hardware Pacing
        current_time = time.time()
        elapsed = current_time - self._last_flush_time
        target_elapsed = 1.0 / self.fps

        # Only sleep if the math took LESS time than the frame budget
        sleep_time = target_elapsed - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        # Record the time AFTER the sleep as the new baseline for the next frame
        self._last_flush_time = time.time()

        # Package the Payload
        # Pack the frame index and the structured dictionary we built via plot(), trace(), etc.
        payload = {
            'frame_index': frame_index,
            'data': self._staging_buffer
        }

        # Dispatch the Payload
        if self.enable_gui and self.queue is not None:
            try:
                # block=True (default) creates natural backpressure.
                # If the GUI is slow, the simulation will pause here and wait.
                self.queue.put(payload)
            except Exception as e:
                print(f"DspProbe: Failed to push to queue: {e}")
        else:
            # Headless mode: Insert datalake / file saving logic here
            pass

        # Reset the Staging Area
        # Assign a brand NEW dictionary rather than calling .clear()
        # This ensures the OS multiprocessing queue pickler doesn't accidentally
        # read a cleared dictionary if it runs slightly asynchronously.
        self._staging_buffer = {}
        self.frame_index += 1

    def _signal_frame(self, signal_in, start_index, windows_per_frame,
                      window_size, ext_samples):
        """
        Extracts a 2D frame of overlapping windows using NumPy stride tricks.

        :param signal_in: 1D NumPy array representing the input signal.
        :param start_index: Starting index for the core frame.
        :param windows_per_frame: Number of rows in the resulting 2D frame.
        :param window_size: Core size of each window before extending.
        :param window_overlap: Multiplier indicating the extension on the left and right
                               (e.g., 1.0 means adding a full window_size to each side).
        :param window_gain
        :return: 2D NumPy array of shape (windows_per_frame, window_size + 2 * extension)
        """

        # Calculate the absolute start and end indices needed from the input array
        # The first row starts at: start_index - ext_samples
        # The last row ends at: start_index + (windows_per_frame * window_size) + ext_samples
        required_start = start_index - ext_samples
        required_end = start_index + (windows_per_frame * window_size) + ext_samples

        # Calculate necessary zero-padding if the required bounds fall outside signal_in
        pad_left = max(0, -required_start)
        pad_right = max(0, required_end - len(signal_in))

        # Safe bounds for slicing the valid part of the input signal
        slice_start = max(0, required_start)
        slice_end = min(len(signal_in), required_end)

        valid_data = signal_in[slice_start:slice_end]

        # Apply zero padding if we hit the boundaries (start or end of the signal)
        if pad_left > 0 or pad_right > 0:
            padded_data = np.pad(valid_data, (pad_left, pad_right), mode='constant', constant_values=0)
        else:
            padded_data = valid_data

        # Use NumPy stride tricks to construct the 2D array
        row_length = window_size + 2 * ext_samples
        shape = (windows_per_frame, row_length)

        # Each row advances by 'window_size' samples relative to the previous row
        strides = (window_size * padded_data.itemsize, padded_data.itemsize)

        # Create a copy of the strided view
        frame_2d = np.copy(as_strided(padded_data, shape=shape, strides=strides, writeable=False))

        # Edge Tapering (Custom Tukey Window)
        taper_len = int(.65 * window_size)

        # Generate the rising half of a Hann window: 0.5 * (1 - cos(pi * n / L))
        taper = 0.5 * (1.0 - np.cos(np.pi * np.arange(taper_len) / taper_len))

        # Build the 1D flat window: [Rising Taper] + [Flat 1.0s] + [Falling Taper]
        custom_window = np.ones(row_length, dtype=np.float32)
        custom_window[:taper_len] = taper
        custom_window[-taper_len:] = taper[::-1]

        # Apply the window to all rows simultaneously via NumPy broadcasting
        frame_2d *= custom_window

        return frame_2d

    def animate_sweep(self, name, signal, seg_size, nrow, fps, loop_duration, gain, start_index, ext_samples):
        """
        Takes a fully computed 1D array and animates a sweep through it over time
        in the GUI, pausing the main simulation thread until the animation completes.
        """
        # If the GUI is disabled, just skip the animation entirely
        if not self.enable_gui:
            return

        ext_samples = int(np.round(ext_samples))

        nframes_to_show = loop_duration * fps
        nframes = len(signal) // (seg_size * nrow)
        frames_stride = nframes // nframes_to_show

        # Initialize the dashboard for this sweep
        self.clear(max_frames=nframes_to_show, fps=fps, loop=False)

        current_index = start_index

        for frame_index in range(nframes_to_show):
            self.plot(
                name=name,
                frame_index=frame_index,  # Assuming your plot signature uses this
                input_signal=signal,
                gain=gain,
                start_index=current_index,
                nrow=nrow,
                seg_size=seg_size,
                ext_samples=ext_samples
            )

            # Flush triggers the dynamic hardware pacing (time.sleep)
            self.flush(0.0)

            # Advance the sweep
            current_index += (frames_stride * nrow * seg_size)
