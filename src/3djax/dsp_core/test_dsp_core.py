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
from utils.plot_sig import *

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

    start_delay = float(config['tx']['start_delay'])
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
    # Convert the list of dicts into a single flat dict
    sequencer_dict = {k: v for d in config['rx']['rx_sequencer'] for k, v in d.items()}

    print("-- Generating Tx Data...")
    tx = Transmitter(config)
    symbols, tx_waveform, tx_analog, sync_block = tx.generate_signal()

    print("-- Apply the Channel...")
    channel = Channel200GHard(config)
    channel_out = channel.process(tx_analog)

    # Generate the local baud rate frequency (oversampled);  this contains
    # the effect of rx frequency offset and jitter (from the yaml configuration).
    fsyn = FrequencySynthesizer(config, len(channel_out))
    rx_in_analog = fsyn.gen_rx_analog(channel_out)

    # Declare a marker to show on plots the state of the receiver (definitions above).
    sm_state = np.zeros(len(rx_in_analog), dtype=np.float16)

    if False:
        # Note on plots: the RX CDR (clock and data recovery) and timing jobs are to take
        # the blue plot (rx analog)  and warp it to recover the red (tx_waveform)
        start_ix = 0
        win_len = 3000
        label_w = 'tx_waveform @ ideal Fs'
        label_tx = 'tx transmitted on the line @ Fs+tx_ppm'
        label_rx = 'rx received from channel @ Fs+rx_ppm'
        title = f"Transmitter/Channel/Receiver with frequency offset and jitter."
        plot_time_domain(start_ix, win_len, tx_waveform,
                         label_w, tx_analog, label_tx, rx_in_analog, label_rx, title)

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
    rx_agc_out[rx_index:] *= digital_rx_gain  # Apply the gain

    if False:
        start_ix = rx_index - 512
        win_len = 4000
        test_timing_and_gains = rx_agc_out[start_ix:start_ix + win_len]
        take_times = np.arange(start_ix, start_ix + win_len, os_factor)
        plot_acq_times(start_ix, win_len, rx_agc_out, 'Rx gains and timing',
                       take_times,
                       title='Acquisition Times')

    #
    # -START OF THE BLOCK PROCESSING - act on rx_agc_out (it is scaled and
    # sampling phase has been coarse corrected
    #

    # Calculate how many full blocks fit from the detection point to the end of the simulation
    remaining_samples = len(channel_out) - sync_index
    num_full_blks = remaining_samples // int(Tblk_ana)
    end_idx = sync_index + (num_full_blks * int(Tblk_ana))

    # Isolate an integer number of Tblk long samples aligned exactly to the detection point
    channel_out_blk = channel_out[sync_index:end_idx]
    # channel_out_blk = channel_out_blk.reshape(num_full_blks, int(Tblk_ana))

    # generate Fout (e.g.100Gbd) for the entire experiment;  this signal would
    # normally be shared by all the lanes,  there is no feedback to it.
    agc_alpha_cold = config['rx']['agc']['agc_alpha_cold']

    # init state machine

    # Start of the block processing

    for blk_idx in range(num_full_blks):
        # AGC
        # DEMUX
        pass

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
