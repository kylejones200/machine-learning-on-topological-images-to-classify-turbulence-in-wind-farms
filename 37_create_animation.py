#!/usr/bin/env python3
import logging

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

np.random.seed(42)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
FPS, N_FRAMES = 10, 100
n_samples = 400

# Generate turbulent wind data
turbulence_intensity = 0.1 + 0.15 * np.sin(2 * np.pi * np.arange(n_samples) / 200)
wind_speed = 8 + 2 * np.sin(2 * np.pi * np.arange(n_samples) / 100)
for i in range(n_samples):
    wind_speed[i] += np.random.normal(
        0, max(0.1, turbulence_intensity[i] * abs(wind_speed[i]))
    )

fig = plt.figure(figsize=(14, 8), facecolor="white")
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
ax1, ax2, ax3 = (
    fig.add_subplot(gs[0, :]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
)

for ax in [ax1, ax2, ax3]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def update(frame):
    ax1.clear()
    ax2.clear()
    ax3.clear()
    end_idx = int((frame / N_FRAMES) * n_samples)

    ax1.plot(wind_speed[:end_idx], "black", linewidth=1.5)
    ax1.set_xlabel("Time", fontsize=10)
    ax1.set_ylabel("Wind Speed (m/s)", fontsize=10)
    ax1.set_title(
        f"Turbulent Wind Speed - Frame {frame + 1}/{N_FRAMES}",
        fontsize=11,
        fontweight="normal",
    )
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    if end_idx > 30:
        window_ti = turbulence_intensity[max(0, end_idx - 30) : end_idx]
        ax2.plot(window_ti, "gray", linewidth=2)
        ax2.set_xlabel("Recent Time", fontsize=10)
        ax2.set_ylabel("Turbulence Intensity", fontsize=10)
        ax2.set_title("Turbulence Intensity", fontsize=11, fontweight="normal")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

    current_ti = turbulence_intensity[min(end_idx - 1, n_samples - 1)]
    classification = (
        "High" if current_ti > 0.2 else "Medium" if current_ti > 0.15 else "Low"
    )
    color = "red" if current_ti > 0.2 else "orange" if current_ti > 0.15 else "green"
    ax3.text(
        0.5,
        0.7,
        f"{classification} Turbulence",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=color,
        transform=ax3.transAxes,
    )
    ax3.text(
        0.5,
        0.4,
        f"TI: {current_ti:.2f}",
        ha="center",
        va="center",
        fontsize=12,
        transform=ax3.transAxes,
    )
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")
    ax3.set_title("Classification", fontsize=11, fontweight="normal")
    return []



def main():
    logger.info("Creating animation for Article 37...")
    anim = animation.FuncAnimation(
        fig, update, frames=N_FRAMES, interval=1000 / FPS, blit=True, repeat=True
    )
    anim.save("37_turbulence_animation.gif", writer="pillow", fps=FPS, dpi=100)
    logger.info("✓ Animation saved: 37_turbulence_animation.gif")
    plt.close()


if __name__ == "__main__":
    main()
