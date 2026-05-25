import threading
import numpy as np
from scipy.signal import fftconvolve
import dearpygui.dearpygui as dpg

from utils.phosphor_engine import PhosphorEngine


class DspGui:
    def __init__(self, config, queue):
        self.config = config
        self.tools_cfg = config.get('tools', {})
        self.queue = queue
        self.is_running = True

        # Extract dimensions from config (handling your exact spelling)
        self.win_res_x = self.tools_cfg.get('win_resolutin_x', 1000)
        self.plot_res_y = self.tools_cfg.get('plot_resolution_y', 700)
        self.trace_res_y = self.tools_cfg.get('trace_resolution_y', 500)
        self.spec_res_y = self.tools_cfg.get('spec_resolution_y', 300)
        self.scalar_res_x = self.tools_cfg.get('scalar_resolution_x', 300)

        # Calculate total viewport dimensions
        self.total_width = self.win_res_x + self.scalar_res_x
        self.total_height = self.plot_res_y + self.trace_res_y + self.spec_res_y

        # Initialize the GPU rendering engine
        self.engine = PhosphorEngine(config)

        # =======================================================================
        # Phosphorus Kernel & Configuration Initialization
        # =======================================================================
        self.enable_phosphor_bloom = self.tools_cfg.get('enable_phosphor_bloom', True)
        self.p31_grid_size_pix = self.tools_cfg.get('p31_grid_size_pix', 31)
        self.core_gaussian_sd = self.tools_cfg.get('core_gaussian_sd', 1.0)
        self.halo_gaussian_sd = self.tools_cfg.get('halo_gaussian_sd', 4.0)

        # Pre-calculate the Triple-Layer Exponential/Gaussian Kernel
        center = self.p31_grid_size_pix // 2
        y, x = np.mgrid[-center:center + 1, -center:center + 1]
        r = np.sqrt(x ** 2 + y ** 2)

        # Layer E: Beam Core (Gaussian)
        E = np.exp(-(r ** 2) / (2.0 * self.core_gaussian_sd ** 2))
        if E.sum() > 0: E /= E.sum()

        # Layer F: Local Bleed (Exponential)
        F = np.exp(-r / (self.halo_gaussian_sd * 0.2))
        if F.sum() > 0: F /= F.sum()

        # Layer G: Atmospheric Glow (Exponential)
        G = np.exp(-r / (self.p31_grid_size_pix / 16.0))
        if G.sum() > 0: G /= G.sum()

        # Blend and normalize
        dot_intens = [1.0, 0.4, 1.0]
        self.phosphor_kernel = (E * dot_intens[0]) + (F * dot_intens[1]) + (G * dot_intens[2])
        self.phosphor_kernel /= self.phosphor_kernel.sum()

        # =======================================================================
        # Pre-calculate the P31 Colormap Lookup Table (Pure NumPy)
        # =======================================================================
        # Matplotlib's hsv_to_rgb([0.45, 0.8, 1.0]) translates exactly to this RGB vector:
        rgb_base = np.array([0.2, 1.0, 0.76], dtype=np.float32)

        map_index = np.linspace(0, 1, 1024)[:, np.newaxis]
        multfactor = 10.0

        cmap_colors = np.tanh(map_index * rgb_base * multfactor)
        cmap_colors = np.clip(cmap_colors, 0.0, 1.0)

        # DearPyGui requires RGBA arrays, so we add a fully opaque Alpha channel
        alpha_channel = np.ones((1024, 1), dtype=np.float32)
        self.cmap_lut = np.hstack((cmap_colors, alpha_channel)).astype(np.float32)

    @classmethod
    def launch(cls, config, queue):
        """
        The Bootstrap Function.
        """
        gui_app = cls(config, queue)
        gui_app._setup_dpg()

        worker_thread = threading.Thread(target=gui_app._processing_loop, daemon=True)
        worker_thread.start()

        gui_app._run_dpg()

    def _setup_dpg(self):
        """
        Builds the 4-region layout and applies the custom themes.
        """
        dpg.create_context()

        # ==========================================
        # Dynamic Texture Setup
        # ==========================================
        mon_x = self.config['tools']['win_resolutin_x']
        mon_y = self.config['tools']['plot_resolution_y']

        # DPG needs a flattened 1D float array of size (Width * Height * 4 RGBA channels)
        blank_texture = np.zeros(mon_x * mon_y * 4, dtype=np.float32)

        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(width=mon_x, height=mon_y, default_value=blank_texture, tag="plot_texture")

        #  Define Themes
        # Global background: Very dark gray.
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (10, 10, 10, 255))
                # Make borders slightly lighter to separate regions
                dpg.add_theme_color(dpg.mvThemeCol_Border, (55, 55, 55, 255))

        # CRT Glass Theme: Very, very dark green for the 3 left windows
        with dpg.theme() as crt_theme:
            with dpg.theme_component(dpg.mvChildWindow):
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (1, 7, 1, 255))

        # Build the Layout
        with dpg.window(tag="MainWindow"):
            # Use a horizontal group to split Left (Signals) and Right (Scalars)
            with dpg.group(horizontal=True):
                # Left stack (3 Stacked Windows)
                with dpg.group():
                    # Plot window (top)
                    with dpg.child_window(tag="PlotWin", width=self.win_res_x, height=self.plot_res_y):
                        # The image widget stretches the texture to fill the UI region
                        dpg.add_image("plot_texture", width=self.win_res_x, height=self.plot_res_y)

                    with dpg.child_window(tag="TraceWin", width=self.win_res_x, height=self.trace_res_y):
                        dpg.add_text("Trace Region (Convergence / Errors)", color=(51, 255, 51))

                    with dpg.child_window(tag="SpecWin", width=self.win_res_x, height=self.spec_res_y):
                        dpg.add_text("Spec Region (Heatmaps / FFE)", color=(51, 255, 51))

                # Right column (Scalars)
                with dpg.child_window(tag="ScalarWin", width=self.scalar_res_x, height=self.total_height):
                    dpg.add_text("Live Metrics", color=(200, 200, 200))
                    dpg.add_separator()

                    # Frame counter
                    dpg.add_text("FRAME: 0", tag="metric_frame", color=(255, 200, 0))

                    dpg.add_text("SNR: -- dB", tag="metric_snr")

        # Apply Themes and Viewport Settings
        dpg.bind_theme(global_theme)

        # Bind the dark green theme specifically to the 3 left windows
        dpg.bind_item_theme("PlotWin", crt_theme)
        dpg.bind_item_theme("TraceWin", crt_theme)
        dpg.bind_item_theme("SpecWin", crt_theme)

        # Make the layout fill the entire OS window without its own scrollbars/title
        dpg.set_primary_window("MainWindow", True)

        dpg.create_viewport(
            title='DSP Monitor Dashboard',
            width=self.total_width + 20,  # Slight padding for OS borders
            height=self.total_height + 40
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()

    # Core Processing & Optics Rendering Methods
    def process_frame(self, frame_2d, window_gain):
        """
        Interpolates X, scales/clips Y, and calculates beam intensity.
        """
        windows_per_frame, original_length = frame_2d.shape
        x_res = self.config['tools']['win_resolutin_x']
        y_res = self.config['tools']['plot_resolution_y']

        if original_length != x_res:
            fft_data = np.fft.fft(frame_2d, axis=1)
            fft_padded = np.zeros((windows_per_frame, x_res), dtype=np.complex128)
            half_len = original_length // 2
            fft_padded[:, :half_len] = fft_data[:, :half_len]

            if original_length % 2 == 0:
                fft_padded[:, half_len] = fft_data[:, half_len] / 2.0
                fft_padded[:, x_res - half_len] = fft_data[:, half_len] / 2.0
                fft_padded[:, x_res - half_len + 1:] = fft_data[:, half_len + 1:]
            else:
                fft_padded[:, x_res - half_len:] = fft_data[:, half_len + 1:]

            upsampled_y = np.real(np.fft.ifft(fft_padded, axis=1)) * (x_res / original_length)
        else:
            upsampled_y = frame_2d.copy()

        scaled_y = upsampled_y * window_gain
        y_max = y_res / 2.0
        frame_pix = np.clip(scaled_y, -y_max, y_max).astype(np.float16)

        dy = np.diff(frame_pix, axis=1)
        dy = np.pad(dy, ((0, 0), (0, 1)), mode='edge')
        distance = np.sqrt(1.0 + dy ** 2)
        frame_intensity = (1.0 / distance).astype(np.float16)

        return frame_pix, frame_intensity

    def render_frame(self, frame_pix, frame_intensity):
        """
        Translates raw coordinates and intensities into a 2D screen matrix.
        Applies FFT convolution for phosphor bloom and opens debug plot.
        """
        x_res = self.config['tools']['win_resolutin_x']
        y_res = self.config['tools']['plot_resolution_y']
        windows = frame_pix.shape[0]

        y_indices = np.clip(np.round(frame_pix + y_res / 2.0), 0, y_res - 1).astype(np.int32)
        x_indices = np.tile(np.arange(x_res), (windows, 1)).astype(np.int32)

        # canvas = np.zeros((y_res, x_res), dtype=np.float32)
        # np.add.at(canvas, (y_indices.flatten(), x_indices.flatten()), frame_intensity.flatten())

        # --- 2. The Vectorized Scatter (Optimized) ---
        # Convert the 2D coordinates into 1D flattened indices
        flat_indices = y_indices.flatten() * x_res + x_indices.flatten()

        # bincount instantly sums all overlapping intensities at C-speed
        flat_canvas = np.bincount(
            flat_indices,
            weights=frame_intensity.flatten(),
            minlength=y_res * x_res
        )

        # Reshape the 1D result back into our 2D screen matrix
        canvas = flat_canvas.reshape((y_res, x_res)).astype(np.float32)

        if self.enable_phosphor_bloom:
            canvas = fftconvolve(canvas, self.phosphor_kernel, mode='same')

        return canvas

    def _processing_loop(self):
        """
        The background worker.
        """
        while self.is_running:
            try:
                # Block and wait for a frame from the Probe
                payload = self.queue.get()

                if payload == "EOF":
                    self.is_running = False
                    break

                # Unpack and Update UI

                # Update the frame counter
                frame_idx = payload.get('frame_index', 0)
                # DPG safely allows updating UI values from this background thread!
                dpg.set_value("metric_frame", f"FRAME: {frame_idx}")

                # Extract the payload data
                staged_data = payload.get('data', {})

                # Route to the GPU processing engine
                if 'plot' in staged_data:
                    for signal_name, signal_info in staged_data['plot'].items():
                        frame_2d = signal_info['frame_2d']
                        gain = signal_info['gain']

                        # The engine handles interpolation, convolution, and LUT mapping in VRAM
                        # and returns a ready-to-draw 1D CPU array.
                        rgba_texture = self.engine.process_and_render(frame_2d, gain)

                        # Instantly push the finished texture to the window
                        dpg.set_value("plot_texture", rgba_texture)

            except Exception as e:
                print(f"GUI Worker Thread Error: {e}")
                self.is_running = False

    def _run_dpg(self):
        """
        Takes over the main thread.
        """
        dpg.start_dearpygui()
        self.is_running = False
        dpg.destroy_context()
