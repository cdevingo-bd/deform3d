"""Plot measured simulation traces. Requires optional matplotlib and numpy."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run(csv, output):
    rows = np.genfromtxt(csv, delimiter=",", names=True, dtype=None, encoding="utf8")
    plt.rcParams.update({"figure.facecolor": "#0b1420", "axes.facecolor": "#111f2e",
                         "text.color": "#edf5ff", "axes.labelcolor": "#c3d3e5",
                         "xtick.color": "#9caec2", "ytick.color": "#9caec2",
                         "axes.edgecolor": "#34465b", "font.size": 11,
                         "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    time = rows["time_s"]
    compression = rows["measured_compression_m"] * 1000
    force = (rows["left_force_N"] + rows["right_force_N"]) * 0.5
    colors = {"close": "#4ad9cd", "hold": "#f2bd59", "release": "#b996ff"}
    # Column phase labels come from the integration fixture, not a fitted curve.
    for phase in np.unique(rows["phase"]):
        mask = rows["phase"] == phase
        if phase in ("settle", "recover"):
            continue
        axes[0].plot(compression[mask], force[mask], label=phase.title(),
                     color=colors.get(phase, "#4ad9cd"), lw=2.3)
    axes[0].set(xlabel="Measured middle-shell compression (mm)",
                ylabel="Mean normal force per gripper pad (N)", title="Squeeze and release response")
    axes[0].legend(facecolor="#111f2e", edgecolor="none", labelcolor="#edf5ff")
    axes[1].plot(time, rows["shell_width_m"] * 1000, color="#4ad9cd", lw=2.3,
                 label="Shell width")
    axes[1].plot(time, rows["pad_gap_m"] * 1000, color="#f2bd59", lw=1.6,
                 linestyle="--", label="Pad gap")
    axes[1].set(xlabel="Simulation time (s)", ylabel="Width / gap (mm)",
                title="Mounted enclosure with rigid and soft contents")
    axes[1].legend(facecolor="#111f2e", edgecolor="none", labelcolor="#edf5ff")
    for ax in axes:
        ax.grid(color="#2c3c4d", alpha=0.5, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("deform3d 0.2  /  Gripper validation fixture", fontsize=20, x=0.06, ha="left")
    fig.text(0.06, 0.055, "MuJoCo 4.0.1 • 157 physical tetrahedra • 1,300 recorded samples\n"
             "Illustrative material; no real specimen calibration. Forces and geometry share the same dynamics timestamp.",
             color="#9caec2", fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.79, bottom=0.23, wspace=0.26)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.csv, args.output)
