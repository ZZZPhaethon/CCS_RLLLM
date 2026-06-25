from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

DAILY_CSV = DATA_DIR / "phase1plus_emitters_capture_rate_profile_daily.csv"
HOURLY_CSV = DATA_DIR / "phase1plus_emitters_capture_rate_profile_hourly.csv"

COLORS = {
    "brevik": "#0072B2",
    "celsio": "#D55E00",
    "yara_sluiskil": "#009E73",
    "total": "#222222",
}

LABELS = {
    "brevik": "Brevik",
    "celsio": "Celsio/Klemetsrud",
    "yara_sluiskil": "Yara Sluiskil",
    "total": "Total",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d0d0d0",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.55,
        }
    )


def read_daily() -> dict[str, list[float]]:
    data = {
        "day": [],
        "brevik": [],
        "celsio": [],
        "yara_sluiskil": [],
        "total": [],
    }
    with DAILY_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data["day"].append(float(row["day_of_year"]))
            data["brevik"].append(float(row["brevik_capture_tpd"]))
            data["celsio"].append(float(row["celsio_capture_tpd"]))
            data["yara_sluiskil"].append(float(row["yara_sluiskil_capture_tpd"]))
            data["total"].append(float(row["total_capture_tpd"]))
    return data


def read_hourly() -> dict[str, list[float]]:
    data = {
        "hour": [],
        "brevik": [],
        "celsio": [],
        "yara_sluiskil": [],
        "total": [],
    }
    with HOURLY_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data["hour"].append(float(row["hour_of_year"]))
            data["brevik"].append(float(row["brevik_capture_tph"]))
            data["celsio"].append(float(row["celsio_capture_tph"]))
            data["yara_sluiskil"].append(float(row["yara_sluiskil_capture_tph"]))
            data["total"].append(float(row["total_capture_tph"]))
    return data


def save(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "pdf"):
        path = FIG_DIR / f"{stem}.{suffix}"
        fig.savefig(path)
        print(path)


def plot_daily() -> None:
    data = read_daily()
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    for key in ("brevik", "celsio", "yara_sluiskil"):
        ax.plot(data["day"], data[key], color=COLORS[key], linewidth=1.9, label=LABELS[key])
    ax.plot(data["day"], data["total"], color=COLORS["total"], linewidth=2.3, label=LABELS["total"])
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Captured CO2 (t/day)")
    ax.set_xlim(1, 365)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    save(fig, "phase1plus_emitters_daily_capture_profile")
    plt.close(fig)


def plot_hourly() -> None:
    data = read_hourly()
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    for key in ("brevik", "celsio", "yara_sluiskil"):
        ax.plot(data["hour"], data[key], color=COLORS[key], linewidth=1.4, label=LABELS[key])
    ax.plot(data["hour"], data["total"], color=COLORS["total"], linewidth=1.9, label=LABELS["total"])
    ax.set_xlabel("Hour of year")
    ax.set_ylabel("Captured CO2 rate (t/h)")
    ax.set_xlim(1, 8760)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    save(fig, "phase1plus_emitters_hourly_capture_profile")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    configure_style()
    plot_daily()
    plot_hourly()


if __name__ == "__main__":
    main()
