import argparse
import yaml
import os
from pathlib import Path
import numpy as np
from transmitter import Transmitter
from channel_200G_hard import Channel200GHard
from analog_front_end import AnalogFrontEnd
from agc import AGC
from frequency_synthesizer import FrequencySynthesizer
from a_phase_interp import APhaseInterp
from adc import ADC
from rx_gains import RxGains
import matplotlib.pyplot as plt

"""  main state machine
0.0 : DETECT_SIG
0.5 : AGC_COLD
1.0 : COARSE_PHASE
1.5 : RX_GAINS
2.0 : TRAINING_FFE
2.5 : TRACKING (MMPD + dPI + FFE continuous loop)
"""


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
    Tsym = 1 / baud_rate
    if 'Tsym' not in config['system']:
        config['system']['Tsym'] = Tsym  # baud rate (e.g. 100GHz)

    Tana = Tsym / os_factor
    if 'Tana' not in config['system']:
        config['system']['Tana'] = Tana  # analog sample duration (e.g. 800GHz)

    digital_streams = int(config['system']['digital_streams'])
    Tdig = Tsym * digital_streams
    if 'Tdig' not in config['system']:
        config['system']['Tdig'] = Tdig  # digital sample duration (e.g. 1.5625GHz)

    digital_block_size = int(config['system']['digital_block_size'])
    Tblk = Tdig * digital_block_size
    if 'Tblk' not in config['system']:
        config['system']['Tblk'] = Tblk  # digital block duration (e.g. 38MHz)

    start_delay = float(config['tx'].get('start_delay', 0.0))
    assert start_delay < Tblk, f"start_delay:{start_delay} >= Tblk:{Tblk}"

    # Integer sample counts for slicing
    Tblk_sym = int(digital_streams * digital_block_size)
    Tblk_ana = int(Tblk_sym * os_factor)

    start_delay =  float(config['tx']['start_delay'])
    assert start_delay < Tblk, f"start_delay:{start_delay} >= Tblk:{Tblk}"

    # Map the string states to the float values for plotting
    STATE_MAP = {
        'SIG_DETECT': 0.0,
        'AGC_COLD': 0.5,
        'COARSE_PHASE': 1.0,
        'RX_GAINS': 1.5,
        'TRAINING_FFE': 2.0,
        'TRACKING_MLSE': 2.5,
        'DATA': 3.0
    }

    print("-- Generating Tx Data...")
    tx = Transmitter(config)
    symbols, tx_analog = tx.generate_signal()

    print("-- Apply the Channel...")
    channel = Channel200GHard(config)
    channel_out = channel.process(tx_analog)

     # Analog Front End (AGC + CTLE)
    print("-- Receiver: AFE...")
    afe = AnalogFrontEnd(config)
    agc = AGC(config)

    # -- SIG_DETECT;
    # Check for signal detection only during the first block
    first_block_out = channel_out[:int(Tblk_ana)]
    rx_afe_first_blk, sd_flag_first_blk, vga_gain_first_blk = afe.process(first_block_out)

    # Check if Squelch broke in the first block
    if not np.any(sd_flag_first_blk > 0.5):
        print(f"CRITICAL: No signal detected in the first block ({Tblk_ana} analog samples).")
        exit(1)

    # Find the exact analog index of detection in the first block
    signal_start_idx = np.argmax(sd_flag_first_blk > 0.5)

    # Calculate how many full blocks fit from the detection point to the end of the simulation
    remaining_samples = len(channel_out) - signal_start_idx
    num_full_blks = remaining_samples // int(Tblk_ana)
    end_idx = signal_start_idx + (num_full_blks * int(Tblk_ana))

    # Isolate an integer number of Tblk long samples aligned exactly to the detection point
    channel_out_blk = channel_out[signal_start_idx:end_idx]
    channel_out_blk = channel_out_blk.reshape(num_full_blks, int(Tblk_ana))

    # -- AGC_COLD

    freq_synth = FrequencySynthesizer(config, len(rx_afe_out))
    rx_sample_times = freq_synth.gen_sample_times(len(rx_afe_out))

    # init state machine
    sm_state = np.zeros(len(rx_afe_out), dtype=np.float32)

    # STATE 0.0 -> 0.5: Wait for Signal & AGC Settle
    signal_start_analog_idx = np.argmax(sd_flag > 0.5)
    sm_state[signal_start_analog_idx:] = 0.5

    print("-- Receiver: Phase Acquisition (Genie Search)...")
    api = APhaseInterp(config)

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
    a = 0


if __name__ == "__main__":
    main()
