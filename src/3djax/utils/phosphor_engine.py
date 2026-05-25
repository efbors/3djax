import cupy as cp
from cupyx.scipy.signal import fftconvolve


class PhosphorEngine:
    def __init__(self, config):
        self.config = config
        self.tools_cfg = config.get('tools', {})

        # =======================================================================
        # Kernel Initialization (Directly on GPU)
        # =======================================================================
        self.enable_phosphor_bloom = self.tools_cfg.get('enable_phosphor_bloom', True)
        self.p31_grid_size_pix = self.tools_cfg.get('p31_grid_size_pix', 31)
        self.core_gaussian_sd = self.tools_cfg.get('core_gaussian_sd', 1.0)
        self.halo_gaussian_sd = self.tools_cfg.get('halo_gaussian_sd', 4.0)

        center = self.p31_grid_size_pix // 2
        grid_1d = cp.arange(-center, center + 1)
        x, y = cp.meshgrid(grid_1d, grid_1d)
        r = cp.sqrt(x ** 2 + y ** 2)

        E = cp.exp(-(r ** 2) / (2.0 * self.core_gaussian_sd ** 2))
        if E.sum() > 0: E /= E.sum()

        F = cp.exp(-r / (self.halo_gaussian_sd * 0.2))
        if F.sum() > 0: F /= F.sum()

        G = cp.exp(-r / (self.p31_grid_size_pix / 16.0))
        if G.sum() > 0: G /= G.sum()

        dot_intens = [1.0, 0.4, 1.0]
        self.phosphor_kernel = (E * dot_intens[0]) + (F * dot_intens[1]) + (G * dot_intens[2])
        self.phosphor_kernel /= self.phosphor_kernel.sum()

        # =======================================================================
        # LUT Initialization (Directly on GPU)
        # =======================================================================
        rgb_base = cp.array([0.2, 1.0, 0.76], dtype=cp.float32)
        map_index = cp.linspace(0, 1, 1024)[:, cp.newaxis]
        multfactor = 10.0

        cmap_colors = cp.tanh(map_index * rgb_base * multfactor)
        cmap_colors = cp.clip(cmap_colors, 0.0, 1.0)

        alpha_channel = cp.ones((1024, 1), dtype=cp.float32)
        self.cmap_lut = cp.hstack((cmap_colors, alpha_channel)).astype(cp.float32)

    def process_and_render(self, frame_2d_np, window_gain):
        """
        Executes the entire optical pipeline on the GPU.
        Returns a 1D flat NumPy array ready for DearPyGui.
        """
        # 1. Pull data into VRAM
        frame_2d = cp.asarray(frame_2d_np)
        windows_per_frame, original_length = frame_2d.shape
        x_res = self.config['tools']['win_resolutin_x']
        y_res = self.config['tools']['plot_resolution_y']

        # 2. Sinc Interpolation (GPU FFT)
        if original_length != x_res:
            fft_data = cp.fft.fft(frame_2d, axis=1)
            fft_padded = cp.zeros((windows_per_frame, x_res), dtype=cp.complex128)
            half_len = original_length // 2
            fft_padded[:, :half_len] = fft_data[:, :half_len]

            if original_length % 2 == 0:
                fft_padded[:, half_len] = fft_data[:, half_len] / 2.0
                fft_padded[:, x_res - half_len] = fft_data[:, half_len] / 2.0
                fft_padded[:, x_res - half_len + 1:] = fft_data[:, half_len + 1:]
            else:
                fft_padded[:, x_res - half_len:] = fft_data[:, half_len + 1:]

            upsampled_y = cp.real(cp.fft.ifft(fft_padded, axis=1)) * (x_res / original_length)
        else:
            upsampled_y = frame_2d

        # 3. Y-Axis Scaling & Distance Intensity
        scaled_y = upsampled_y * window_gain
        y_max = y_res / 2.0
        frame_pix = cp.clip(scaled_y, -y_max, y_max).astype(cp.float16)

        dy = cp.diff(frame_pix, axis=1)
        dy = cp.pad(dy, ((0, 0), (0, 1)), mode='edge')
        distance = cp.sqrt(1.0 + dy ** 2)
        frame_intensity = (1.0 / distance).astype(cp.float16)

        # 4. Vectorized Scatter (cp.bincount)
        y_indices = cp.clip(cp.round(frame_pix + y_res / 2.0), 0, y_res - 1).astype(cp.int32)
        x_indices = cp.tile(cp.arange(x_res), (windows_per_frame, 1)).astype(cp.int32)

        flat_indices = y_indices.flatten() * x_res + x_indices.flatten()
        flat_canvas = cp.bincount(flat_indices, weights=frame_intensity.flatten(), minlength=y_res * x_res)
        canvas = flat_canvas.reshape((y_res, x_res)).astype(cp.float32)

        # 5. Phosphor Bloom (GPU FFT Convolution)
        if self.enable_phosphor_bloom:
            canvas = fftconvolve(canvas, self.phosphor_kernel, mode='same')

        # 6. Gamma Correction & Colormap LUT Mapping
        vmax = cp.max(canvas) * 0.4
        if vmax > 0:
            norm_canvas = cp.clip(canvas / vmax, 0.0, 1.0)
        else:
            norm_canvas = canvas

        norm_canvas = norm_canvas ** 0.8
        # .astype() ensures the type conversion happens directly on the CUDA cores
        lut_indices = cp.clip((norm_canvas * 1023).astype(cp.int32), 0, 1023)

        rgba_image = self.cmap_lut[lut_indices]

        # 7. Push flattened texture back to System RAM for DPG
        return cp.asnumpy(rgba_image.flatten())