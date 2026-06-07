from dsp_core.trasmitter_qam import PhysicalTransmitter
from channel.channel_refdisres import ChannelRefDisRes
from dsp_core.rx_afe import RxAFE
import numpy as np


class CoherentPipeline:
    def __init__(self, config, batch_size):
        self.config = config
        self.batch_size = batch_size

        self.am_symbols_fadc = 2* self.config['system'].get('am_length_symbols')
        # Establish symmetric physical layer hardware modules
        self.tx = PhysicalTransmitter(config)
        self.channel = ChannelRefDisRes(config, B=batch_size)
        self.rx_afe = RxAFE(config)

    def process_batch(self, payload_bits, enable_invariance=False):
        """
        Orchestrates end-to-end signal processing.

        Returns:
            rx_adc_a: Quantized 200 GSa/s array via anchor channel (h_a)
            rx_adc_p: Quantized 200 GSa/s array via invariant channel (h_p) or None
            h_a: The anchor channel impulse response matrix
            h_p: The invariant channel impulse response matrix or None
        """

        # Transmitter
        self.am_ref, self.tx_analog = self.tx.transmit_batch(
            batch_size=self.batch_size,
            payload_bits=payload_bits
        )

        # Synthesize physical channels for this specific transmission step
        self.h_a, self.h_p = self.channel.generate_batch()
        # Process main, Anchor Track (A)
        self.channel_out_a = self.channel.process(self.tx_analog, self.h_a)

        self.rx_adc_a, self.am_start_indices_fx_a = self.rx_afe.process(self.am_ref, self.channel_out_a)

        # Process second, Invariant Track (P) if requested
        if enable_invariance:
            self.channel_out_p = self.channel.process(self.tx_analog, self.h_p)
            self.rx_adc_p, self.am_start_indices_fx_p = self.rx_afe.process(self.am_ref, self.channel_out_p)
        else:
            self.h_p = None
            self.rx_adc_p = None
            self.am_start_indices_fx_p = None

        return

    def get_rx_am_a(self):
        # Get the batch dimension
        batch_size = self.rx_adc_a.shape[0]

        # Create the row indexing vector: shape (n_batch, 1)
        row_idx = np.arange(batch_size)[:, np.newaxis]

        # Create the column extraction grid: shape (n_batch, am_symbols)
        # Adds the base [0, 1, ..., am_len-1] array to each row's unique start index
        col_idx = self.am_start_indices_fx_a[:, np.newaxis] + np.arange(self.am_symbols_fadc)

        # Use advanced integer indexing to extract all rows simultaneously
        rx_am_a = self.rx_adc_a[row_idx, col_idx]

        return rx_am_a
