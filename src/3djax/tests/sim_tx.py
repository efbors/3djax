import argparse
import yaml
import os
from pathlib import Path

from dsp_core.trasmitter_qam import PhysicalTransmitter
from channel.channel_refdisres import ChannelRefDisRes
from utils.plot_sig import plot_channel_batch
from dsp_core.receiver_analog_front_end import ReceiverAnalogFrontEnd


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

    print("-- Generating Tx Data...")

    tx = PhysicalTransmitter(config)

    tx_analog = tx.transmit_frame(batch_size=batch_size, payload_bits=payload_bits)

    if False:
        # ============================================
        # PLOT Time-Domain Analog Waveforms (Continuous-Time)
        import matplotlib.pyplot as plt
        import numpy as np

        sig_analog = tx_analog[0, :].astype(np.complex64)
        plt.plot(np.real(sig_analog), 'r')
        plt.plot(np.imag(sig_analog), 'g')
        plt.plot(np.abs(sig_analog), 'b')
        plt.grid()
        plt.show(block=False)

    if False:
        # ===========================================
        # PLOT 2: Constellation Output - 8-Phase "Lazy" Downsampling Grid
        import matplotlib.pyplot as plt
        import numpy as np
        sig_analog = tx_analog[0, :].astype(np.complex64)
        # Assuming sig_analog is your 800 GSa/s complex64 waveform
        sig_I = np.real(sig_analog)
        sig_Q = np.imag(sig_analog)

        # Undo the OQAM shift (Shift Q backwards by OS // 2)
        # Using np.roll shifts the data without changing the array length
        sig_Q_realigned = np.roll(sig_Q, shift=-4)

        # Recombine into a standard QAM complex signal
        sig_analog_realigned = sig_I + 1j * sig_Q_realigned

        print("-- Plotting 8-phase downsampling grid...")
        # We use a large figure size for the 2x4 grid to keep points legible
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
        fig.suptitle(f"Transmitter Constellation: Downsampling Grid (OS={os_factor})\n"
                     f"{config['system']['modulation']} with OQAM shift, Filtering, and Non-linearities", fontsize=16)

        # Discard the Alignment Marker (first 256 symbols) to look only at the payload
        payload_start_sample = 256 * os_factor
        payload_signal = sig_analog_realigned[payload_start_sample:]

        # Iterate through all 8 possible sampling offsets (0 to 7)
        for offset in range(os_factor):
            # Determine subplot indices (2 rows, 4 columns)
            row = offset // 4
            col = offset % 4
            ax = axes[row, col]

            # "Lazy" Downsampling: Slice the 8x array, starting at 'offset', stepping by 'os_factor'
            # This simulates 8 different receiver clock phases.
            sig_baud = payload_signal[offset::os_factor]

            # Plot standard constellation (Scatter real vs imag)
            # We use a very small alpha and marker size because there are thousands of points.
            ax.scatter(np.real(sig_baud), np.imag(sig_baud),
                       s=1.0, alpha=0.1, color='purple')

            # Aesthetics for the subplot
            ax.set_title(f"Sampling Offset: {offset}")
            ax.grid(True, which='both', linestyle=':', alpha=0.3)
            ax.set_aspect('equal', 'box')  # Force circular shape

            # Draw reference axes lines through (0,0)
            ax.axhline(0, color='black', lw=0.5, alpha=0.5)
            ax.axvline(0, color='black', lw=0.5, alpha=0.5)

            # Set rigid axis limits based on expected QAM power + PAPR room
            # For a normalized QAM16 (max amp ~1.26), +/- 2.0 covers clipping and impairments.
            ax.set_xlim([-2.0, 2.0])
            ax.set_ylim([-2.0, 2.0])

            # Add labels only to edge plots to reduce clutter
            if row == 1: ax.set_xlabel("In-Phase (I)")
            if col == 0: ax.set_ylabel("Quadrature (Q)")

        print("-- Analysis Complete. Looking at Plot 2:")
        print(f"   Since you are using Offset QAM (OQAM) and severe 60GHz filtering,\n"
              f"   NONE of these plots will look like a perfect standard QAM grid.\n"
              f"   The OQAM shift means I and Q are offset by T/2. You must realign\n"
              f"   them (shift Q by 4 samples) in the receiver BEFORE these plots make sense.")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for main title
        plt.show()  # Final block show

    # probe = DspProbe(config)

    if False:
        probe.animate_sweep('tx_analog', tx_analog, seg_size=os_factor, nrow=64, fps=30,
                            loop_duration=10, gain=60.0, start_index=1000, ext_samples=1 * os_factor)

    print("-- Generating Channel Impulse Responses (Anchor & Positive Pairs)...")
    channel = ChannelRefDisRes(config, B=batch_size)

    # Generate the batch of Anchors and their micro-drifted Positive Pairs
    h_a, h_p = channel.generate_batch()

    if False:
        # Plot the anchors to verify the macro variance (different trace lengths, etc.)
        # print("-- Plotting Anchor Channels...")
        # plot_channel_batch(h_a, channel.dt)
        # plot_channel_batch(h_p, channel.dt)
        # intrinsic dimension 260531
        # channel.estimate_h_dimension(h_a)
        # TwoNN Estimated ID: 8.96 (Took 6.46s)
        # MLE Estimated ID:   8.58 (Took 1.89s)
        # set the Embedded Space size = 16;
        pass

    channel_out = channel.process(tx_analog, h_a)
    # channel_out = channel.process(tx_analog, h_p)

    if True:
        import matplotlib.pyplot as plt
        import numpy as np
        plt.plot(np.real(channel_out[0]))
        plt.grid()
        plt.show(block=False)

    rx_afe = ReceiverAnalogFrontEnd(config)
    rx_adc_out = rx_afe.process(channel_out)

    breakpoint()


if __name__ == "__main__":
    main()
