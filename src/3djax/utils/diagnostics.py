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


import matplotlib.pyplot as plt
import numpy as np


def oqam_eye(signal_in, os_factor, title=''):
    """Undoes OQAM delay on Q and plots a single constellation canvas with a time-colored trend."""
    sig_I = np.real(signal_in)
    sig_Q = np.imag(signal_in)

    # Undo OQAM shift: Roll Q backwards by T/2 (OS // 2) along the sample axis
    sig_Q_realigned = np.roll(sig_Q, shift=-(os_factor // 2), axis=-1)
    sigal = sig_I + 1j * sig_Q_realigned

    # Keep your preferred fig, ax style but setup a single square canvas
    fig, ax = plt.subplots(figsize=(9, 8))  # Widened slightly to accommodate colorbar comfortably
    fig.suptitle(title, fontsize=14)

    # Flatten the arrays to safely handle 2D batch frames or 1D arrays alike
    x_pts_os = np.real(sigal).ravel()
    y_pts_os = np.imag(sigal).ravel()

    x_pts = x_pts_os[::os_factor]
    y_pts = y_pts_os[::os_factor]

    # Create a normalized sequence array tracking the symbol timeline
    symbol_timeline = np.arange(len(x_pts))

    # Map color 'c' to the timeline. 'plasma' transitions  from dark purple to hot pink/yellow
    sc = ax.scatter(x_pts, y_pts, c=symbol_timeline, cmap='plasma', s=4.0, alpha=0.8)

    # Add a proportional colorbar indicating the forward march of time
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Symbol Sequence (Oldest → Newest)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # Calculate dynamic uniform boundaries with 15% extra margin
    max_val = max(np.max(np.abs(x_pts)), np.max(np.abs(y_pts)))
    lim = max_val * 1.15

    # Apply symmetric limits
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])

    # Styling details
    ax.grid(True, which='both', linestyle=':', alpha=0.3)
    ax.set_aspect('equal', 'box')
    ax.axhline(0, color='black', lw=0.5, alpha=0.5)
    ax.axvline(0, color='black', lw=0.5, alpha=0.5)
    ax.set_xlabel("In-Phase (I)", fontsize=11)
    ax.set_ylabel("Quadrature (Q)", fontsize=11)

    plt.tight_layout()
    plt.show(block=False)
