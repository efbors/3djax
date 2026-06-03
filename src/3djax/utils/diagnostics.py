import numpy as np
import matplotlib.pyplot as plt


def plot_time_domain(signal_800g, title="Time-Domain Waveform"):
    """Plots raw continuous real, imag, and magnitude vectors."""
    sig = signal_800g[0, :].astype(np.complex64)
    plt.figure(figsize=(10, 4))
    plt.plot(np.real(sig), 'r', label='Real (I)', alpha=0.7)
    plt.plot(np.imag(sig), 'g', label='Imag (Q)', alpha=0.7)
    plt.plot(np.abs(sig), 'b', label='Magnitude (Abs)', alpha=0.5)
    plt.title(title)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    plt.show(block=False)


def plot_oqam_constellation_grid(signal_800g, modulation='QAM16', os_factor=8):
    """Undoes OQAM delay on Q and plots the 8-phase downsampling grid."""
    sig_I = np.real(signal_800g[0, :])
    sig_Q = np.imag(signal_800g[0, :])

    # Undo OQAM shift: Roll Q backwards by T/2 (OS // 2)
    sig_Q_realigned = np.roll(sig_Q, shift=-(os_factor // 2))
    sig_realigned = sig_I + 1j * sig_Q_realigned

    # Strip the Alignment Marker (first 256 symbols) to view pure payload
    payload_signal = sig_realigned[(256 * os_factor):]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    fig.suptitle(f"Transmitter Constellation: Downsampling Grid (OS={os_factor})\n"
                 f"{modulation} Realigned OQAM", fontsize=14)

    for offset in range(os_factor):
        row, col = offset // 4, offset % 4
        ax = axes[row, col]

        # Slicing down to symbol rate based on sample phase offset
        sig_baud = payload_signal[offset::os_factor]
        ax.scatter(np.real(sig_baud), np.imag(sig_baud), s=1.0, alpha=0.1, color='purple')

        ax.set_title(f"Sampling Offset: {offset}")
        ax.grid(True, which='both', linestyle=':', alpha=0.3)
        ax.set_aspect('equal', 'box')
        ax.axhline(0, color='black', lw=0.5, alpha=0.5)
        ax.axvline(0, color='black', lw=0.5, alpha=0.5)
        ax.set_xlim([-2.0, 2.0])
        ax.set_ylim([-2.0, 2.0])

        if row == 1: ax.set_xlabel("In-Phase (I)")
        if col == 0: ax.set_ylabel("Quadrature (Q)")

    plt.tight_layout()
    plt.show(block=False)