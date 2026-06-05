import numpy as np
from dsp_core.trasmitter_qam import PhysicalTransmitter
from channel.channel_refdisres import ChannelRefDisRes
from dsp_core.rx_afe import RxAFE
from utils.diagnostics import oqam_eye


class CoherentPipeline:
    def __init__(self, config, batch_size):
        self.config = config
        self.batch_size = batch_size

        # Establish symmetric physical layer hardware modules
        self.tx = PhysicalTransmitter(config)
        self.channel_medium = ChannelRefDisRes(config, B=batch_size)
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
        self.am_ref, tx_analog = self.tx.transmit_batch(
            batch_size=self.batch_size,
            payload_bits=payload_bits
        )

        # Synthesize physical channels for this specific transmission step
        h_a, h_p = self.channel_medium.generate_batch()
        # Process main, Anchor Track (A)
        channel_out_a = self.channel_medium.process(tx_analog, h_a)

        rx_adc_a = self.rx_afe.process(self.am_ref, channel_out_a)

        # Process second, Invariant Track (P) if requested
        if enable_invariance:
            channel_out_p = self.channel_medium.process(tx_analog, h_p)
            rx_adc_p = self.rx_afe.process(self.am_ref, channel_out_p)
        else:
            h_p = None
            rx_adc_p = None

        return rx_adc_a, rx_adc_p, h_a, h_p
