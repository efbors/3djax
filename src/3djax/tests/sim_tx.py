import argparse
import yaml
import os
from pathlib import Path

from dsp_core.trasmitter_qam import PhysicalTransmitter
from channel.channel_refdisres import ChannelRefDisRes
from utils.plot_sig import plot_channel_batch


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

    channel_out = channel.process(tx_analog, h_a, h_p)
    if False:
        import matplotlib.pyplot as plt
        import numpy as np
        plt.plot(np.real(channel_out[0]))
        plt.grid()
        plt.show(block=False)

    # Generate the local baud rate frequency (oversampled);  this contains
    # the effect of rx frequency offset and jitter (from the yaml configuration).
    fsyn = FrequencySynthesizer(config, len(channel_out))
    rx_in_analog = fsyn.gen_rx_analog(channel_out)

    if True:
        probe.animate_sweep('rx_in_analog', rx_in_analog, seg_size=os_factor, nrow=64, fps=30,
                            loop_duration=10, gain=60.0, start_index=1000, ext_samples=1.5 * os_factor)

    # Declare a marker to show on plots the state of the receiver (definitions above).
    sm_state = np.zeros(len(rx_in_analog), dtype=np.float16)

    agc = AGC(config)

    #
    # --- Signal Detect
    #  look only during the first block (simulation constraint)
    #

    first_block_out = rx_in_analog[:int(Tblk_ana)]
    agc_alpha_hot = config['rx']['agc']['agc_alpha_hot']
    agc.set_alpha(agc_alpha_hot)
    rx_agc_out, gain_hist = agc.process(first_block_out)
    sync_index, matched_filt_out = agc.detect_sync_sequence(rx_agc_out, os_factor, sync_block)
    # alignment marker detected, cool the AGC;
    # note: sync_index marks the end of the detection (the end of the second
    # sync marker)
    agc_cool_t = sequencer_dict['AGC_COLD']
    sm_state[sync_index:] = STATE_MAP['AGC_COLD']
    agc_alpha_cold = config['rx']['agc']['agc_alpha_cold']
    agc.set_alpha(agc_alpha_cold)
    rx_index = sync_index  # move the cursor to the end of the sync bloc

    # Freeze AGC; reprocess the AGC from the new cursor with frozen AGC
    current_agc_gain = agc.get_current_gain()
    rx_agc_out = rx_in_analog * current_agc_gain

    if False:
        start_ix = 0
        win_len = 10000
        plot_time_domain(start_ix, win_len, rx_in_analog, 'Rx in',
                         rx_agc_out, 'rx_agc_out',
                         # gain_hist, 'gains',
                         # 0.1 * matched_filt_out, 'matched filter out',
                         # sm_state, 'state',
                         title='Detect sync')

    #
    # --- Coarse Phase Alignment
    #

    api = APhaseInterp(config)  # analog phase interpolator
    # Run the vector math to find the best UI fractional offset
    pi_eval_block_size = config['rx']['api']['pi_eval_block_size']
    coarse_phase_seg = rx_agc_out[rx_index:rx_index + pi_eval_block_size * os_factor]

    # Get the optimal fractional analog offset
    variances, best_offset = api.calc_coarse_phase(coarse_phase_seg)
    # Round the fractional offset to the nearest integer analog sample
    rx_index += int(np.round(best_offset))  # adjust cursor to the best eye opening

    if False:
        start_ix = rx_index
        win_len = 500
        take_times = np.arange(start_ix, start_ix + win_len, os_factor)
        plot_acq_times(start_ix, win_len, rx_agc_out, 'Rx in',
                       take_times,
                       title='Acquisition Times')

    #
    # -- RX GAINS
    # calculate the post ADC gains to land the levels on the reference
    # PAM4 levels

    rxg = RxGains(config)
    rxg_block_len = 200  # samples at baud rate
    rxg_block = rx_agc_out[rx_index:rx_index + rxg_block_len * os_factor]

    # Calculate optimal post-ADC digital gain
    digital_rx_gain = rxg.calc_post_adc_gain(rxg_block)
    rx_index += rxg_block_len * os_factor

    if False:
        start_ix = rx_index - 512
        win_len = 4000
        test_timing_and_gains = rx_agc_out[start_ix:start_ix + win_len]
        take_times = np.arange(start_ix, start_ix + win_len, os_factor)
        plot_acq_times(start_ix, win_len, rx_agc_out, 'Rx gains and timing',
                       take_times,
                       title='Acquisition Times')

    rx_in = digital_rx_gain * rx_agc_out[rx_index:]  # Apply the gain

    if True:
        probe.animate_sweep('rx_in', signal=rx_in, seg_size=os_factor, nrow=64, fps=30,
                            loop_duration=10, gain=60.0, start_index=1000, ext_samples=1.5 * os_factor)

    #
    # -START OF THE BLOCK PROCESSING - act on rx_agc_out (it is scaled and
    # sampling phase has been coarse corrected
    #

    # approximate of the end of the simulation; because the analog phase interpolator
    # can advance beyond the existing rx samples
    adc = ADC(config)
    ffe = FFE(config)
    timing = Timing(config)
    remaining_samples = len(channel_out) - sync_index
    num_full_blks = remaining_samples // int(Tblk_ana)
    timing_offset = 0.0

    block_len = digital_streams * digital_block_size

    for blk_idx in range(num_full_blks):
        # analog phase interpolator (extract fractionally interpolated samples)
        rx_block_api = api.sample_block(rx_in, rx_index, timing_offset, block_len, os_factor)

        # ADC - quantize
        ffe_in = adc.process(rx_block_api)

        # Note: no need for demuxing for the reference code; this is an optimization for later
        # demux (1:64) - round robin;  adjacent rows receive adjacent samples in time;
        # rx_ffe_in = np.reshape(rx_blk_quant, (digital_block_size, digital_streams)).T

        # -- FFE; calculate both phase error (for a_phase_interap) and the shortened
        # channel for the Viterbi MLSE
        ffe_out, phase_error, com_error = ffe.calc(ffe_in)

        # -- Update timing
        timing_offset = timing(phase_error, com_error)

    # STATE 0.0 -> 0.5: Wait for Signal & AGC Settle
    signal_start_analog_idx = np.argmax(sd_flag > 0.5)
    sm_state[signal_start_analog_idx:] = 0.5

    print("-- Receiver: Phase Acquisition (Genie Search)...")

    # Find the exact analog index where the dead-air ends and the signal begins
    # (sd_flag is 0.75 when active, so we look for > 0.5)
    signal_start_analog_idx = np.argmax(sd_flag > 0.5)

    # Add a settling margin to let the VGA transient decay (e.g., 1000 UI)
    agc_settling_margin = 1000 * api.os_factor
    settled_analog_idx = signal_start_analog_idx + agc_settling_margin

    # STATE 0.5 -> 1.0: Coarse Phase Acquisition
    sm_state[int(settled_analog_idx):] = 1.0

    # Find the first symbol in the generated clock array that happens AFTER the AGC is settled
    first_eval_symbol_idx = np.argmax(rx_sample_times > settled_analog_idx)

    # Pluck the exact block of symbol times for the Genie to evaluate
    start_sym = first_eval_symbol_idx
    end_sym = start_sym + api.pi_eval_block_size + 1
    eval_sample_times = rx_sample_times[start_sym:end_sym]

    # Run the vector math to find the best UI fractional offset
    variances, best_phase_offset = api.calc_coarse_phase(eval_sample_times, rx_afe_out)
    print(f"   -> AGC settled at analog index: {settled_analog_idx}")
    print(f"   -> Found optimal Genie phase offset: +{best_phase_offset:.4f} indices")

    # Apply the winning phase shift to the ENTIRE continuous clock grid!
    rx_sample_times += best_phase_offset

    # --- INSERTED BLOCK START: MASTER SAMPLER, ADC, RX GAINS ---
    # The continuous digital stream using the final clock grid
    cs = CubicSpline(np.arange(len(rx_afe_out)), rx_afe_out)
    rx_symbols_continuous = cs(rx_sample_times)

    # STATE 1.0 -> 1.5: RX Gains (Histogram)
    hist_end_sym = end_sym + 5000
    hist_end_analog_idx = int(rx_sample_times[hist_end_sym])
    sm_state[int(rx_sample_times[end_sym]):] = 1.5

    print("-- Receiver: ADC Quantization and Rx Gains...")
    adc = ADC(config)
    rx_gains_block = RxGains(config)

    # Feed the block immediately following the Genie eval through the ADC
    adc_eval_block = adc.process(rx_symbols_continuous[end_sym:hist_end_sym])

    # Calculate optimal post-ADC digital gain (and plot the histogram!)
    digital_rx_gain = rx_gains_block.calc_post_adc_gain(adc_eval_block, show_plot=True)
    print(f"   -> Calculated Post-ADC Digital Gain: {digital_rx_gain:.4f}")

    # STATE 1.5 -> 2.0: Ready for FFE Training
    sm_state[hist_end_analog_idx:] = 2.0


if __name__ == "__main__":
    main()
