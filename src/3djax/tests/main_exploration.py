import argparse
import yaml
import os
from pathlib import Path
from dsp_core.pipeline import CoherentPipeline
from dsp_core.channel_estimator import ChannelEstimator
from dsp_core.mmse_sir import MmseSir
from dsp_core.bcjr_decoder import BcjrDecoder


def main():
    # Parse CLI Arguments
    parser = argparse.ArgumentParser(description="200G DSP simulation")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Path to the experiment YAML config file")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()

    with open(config_path, 'r') as f:
        raw_yaml = f.read()
        expanded_yaml = os.path.expandvars(raw_yaml)  # Expand all env vars
        config = yaml.safe_load(expanded_yaml)

    # Derive the simulation system parameters (Your exact code)
    os_factor = int(config['system']['os_factor'])
    baud_rate = float(config['system']['baud_rate'])
    batch_size = int(config['system']['batch_size'])
    payload_bits = int(config['system']['payload_bits'])

    Tsym = 1 / baud_rate
    if 'Tsym' not in config['system']:
        config['system']['Tsym'] = Tsym  # baud rate (e.g. 100GHz)

    # ==========================================
    # HARDWARE SIMULATION
    # ==========================================

    # Setup simulation constraints
    batch_size = 64
    num_payload_bits = 1944

    # Initialize the end-to-end coherent transceiver link
    link = CoherentPipeline(config, batch_size=batch_size)

    # Execute the signal path (Keep invariance off for rapid DSP bring-up)
    link.process_batch(num_payload_bits, enable_invariance=False)

    # ==========================================
    # RECEIVER DSP (Channel Estimation)
    # ==========================================
    rx_am_a = link.get_rx_am_a()
    am_ref = link.am_ref

    if False:
        import scipy.signal as signal
        import numpy as np
        import matplotlib.pyplot as plt
        am_len = am_ref.shape[-1]
        am_ref_fsim = np.zeros(am_len * os_factor, dtype=np.complex64)
        am_ref_fsim[::os_factor] = am_ref

        tx_an = link.tx_analog[0]
        corr_tx = signal.correlate(tx_an, am_ref_fsim, mode='same')

        plt.figure(figsize=(10, 4))
        plt.plot(np.abs(corr_tx))
        plt.title("Tap 1: Tx Analog Impulse Response (800 GHz)")
        plt.grid(True)
        plt.show()

        # 2. Tap 2: The Raw Channel Output
        corr_chan = signal.correlate(link.channel_out_a[0], am_ref_fsim, mode='same')

        plt.figure(figsize=(10, 4))
        plt.plot(np.abs(corr_chan))
        plt.title("Tap 2: Channel Output Impulse Response (800 GHz)")
        plt.grid(True)
        plt.show()

    # Run the MMSE equivalent channel estimator
    estimator = ChannelEstimator()

    h_est_zf, snr_linear_est = estimator.estimate_channel_fd_zf(rx_am_a, am_ref, tap_count=2003)
    # h_est_td = estimator.estimate_channel_td_corr(rx_am_a, am_ref)
    # h_est_ls = estimator.estimate_channel_td_ls(rx_am_a, am_ref, tap_count=2003)

    if False:
        import matplotlib.pyplot as plt
        import numpy as np

        if True:
            h_est, title = h_est_zf, 'impulse response (ZF)'
            plt.figure(figsize=(10, 4))
            plt.plot(np.real(h_est[0]), 'r', label='Real', alpha=0.7)
            plt.plot(np.imag(h_est[0]), 'g', label='Imag', alpha=0.7)
            # plt.plot(np.abs(h_est[0]), 'b', label='Magnitude (Abs)', alpha=0.7)
            plt.title(title)
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.legend()
            plt.show(block=False)

        if False:
            # Note: TD has ringing from the shorter correlation window
            h_est, title = h_est_td, 'impulse response (TD)'
            plt.figure(figsize=(10, 4))
            plt.plot(np.real(h_est[0]), 'r', label='Real', alpha=0.7)
            plt.plot(np.imag(h_est[0]), 'g', label='Imag', alpha=0.7)
            plt.plot(np.abs(h_est[0]), 'b', label='Magnitude (Abs)', alpha=0.7)
            plt.title(title)
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.legend()
            plt.show(block=False)

        # h_est, title = h_est_ls, 'impulse response (LS)'

    ffe = MmseSir(config)
    # calculate the FFE taps (w) and the channel

    decoder = BcjrDecoder(config)
    llr = decoder.get_llrs()

    # TODO hook up to eyebox

    breakpoint()


if __name__ == "__main__":
    main()
