from pathlib import Path

import matplotlib.pyplot as plt


def plot_anomaly_timeline(
    flight_data,
    detected_events,
    output_path,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 9),
        sharex=True,
    )

    # Selected flight signals
    axes[0].plot(
        flight_data["time_s"],
        flight_data["ias_kt"],
    )
    axes[0].set_ylabel("IAS [kt]")
    axes[0].set_title(
        "Final Unseen Flight — Hybrid Anomaly Detection"
    )

    axes[1].plot(
        flight_data["time_s"],
        flight_data["roll_deg"],
    )
    axes[1].set_ylabel("Roll [deg]")

    axes[2].plot(
        flight_data["time_s"],
        flight_data["engine_rpm"],
    )
    axes[2].set_ylabel("Engine RPM")
    axes[2].set_xlabel("Time [s]")

    # Ground-truth anomaly regions
    true_mask = flight_data["anomaly"] == 1

    for ax in axes:
        ax.fill_between(
            flight_data["time_s"],
            0,
            1,
            where=true_mask,
            transform=ax.get_xaxis_transform(),
            alpha=0.12,
            label="Ground truth",
        )

        # Reported anomaly events
        for _, event in detected_events.iterrows():
            ax.axvspan(
                event["start_s"],
                event["end_s"],
                alpha=0.18,
            )

        ax.grid(True, alpha=0.3)

    axes[0].legend(
        ["Signal", "Ground truth", "Reported event"],
        loc="upper right",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)