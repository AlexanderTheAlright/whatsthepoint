# The data in this file are simulated for illustration; no group, site or measure is real.
"""Draw the sample figures in docs/samples/ with the default 'house' style.

Run from the repository root:  python docs/sim_make_samples.py

Every figure is checked for overlap and clipping before it is saved, as PNG
(300 dpi) and SVG.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from whatsthepoint.utils.layout import assert_clean, finish
from whatsthepoint.utils.style import BW_LINESTYLES, BW_MARKER_FILLS, BW_MARKERS, save_figure, set_style
from whatsthepoint.visuals.bar import bar
from whatsthepoint.visuals.boxplot import boxplot
from whatsthepoint.visuals.coef import coef
from whatsthepoint.visuals.cooccur import cooccur
from whatsthepoint.visuals.dag import dag
from whatsthepoint.visuals.density import kde
from whatsthepoint.visuals.scatter import scatterplot
from whatsthepoint.visuals.trend import trend

OUT = Path(__file__).parent / "samples"
rng = np.random.default_rng(20240)


# grouped bars with value labels: Group A rises across rounds, Group B stays level
shares = pd.DataFrame({
    "Round": ["Round 1", "Round 2", "Round 3", "Round 4"],
    "Group A": [31.5, 38.0, 46.5, 57.0],
    "Group B": [29.0, 30.5, 28.5, 31.0],
})


def bar_grouped():
    return bar(shares, x="Round", y="Share agreeing (%)", orientation="grouped")[0]


# coefficient plot: three estimates clear of zero, two not
estimates = pd.DataFrame({
    "term": ["Predictor one", "Predictor two", "Predictor three", "Predictor four", "Predictor five"],
    "estimate": [0.42, 0.21, 0.04, -0.09, -0.33],
    "conf.low": [0.27, 0.06, -0.10, -0.25, -0.50],
    "conf.high": [0.57, 0.36, 0.18, 0.07, -0.16],
    "p.value": [0.0001, 0.006, 0.58, 0.27, 0.0002],
})


def coef_intervals():
    return coef(estimates, p_col="p.value")[0]


# trend lines: Site A overtakes Site B, which holds level; Site C declines
periods = np.arange(1, 9)
series = pd.DataFrame({
    "Period": np.tile(periods, 3),
    "Site": np.repeat(["Site A", "Site B", "Site C"], len(periods)),
    "Index score": np.r_[
        38 + 4.0 * (periods - 1) + rng.normal(0, 0.25, len(periods)),
        52 + 0.0 * (periods - 1) + rng.normal(0, 0.25, len(periods)),
        47 - 2.0 * (periods - 1) + rng.normal(0, 0.25, len(periods)),
    ].round(1),
})


def trend_lines():
    return trend(series, x="Period", y="Index score", group="Site")[0]


# two distributions with different centres and spreads
scores = pd.DataFrame({
    "Group": np.repeat(["Group A", "Group B"], 300),
    "Outcome score": np.r_[rng.normal(48, 7, 300), rng.normal(62, 10, 300)],
})


def density_groups():
    return kde(scores, x="Outcome score", group="Group")[0]


# box plots: the median rises and the spread widens from Level 1 to Level 4
levels = pd.DataFrame({
    "Level": np.repeat(["Level 1", "Level 2", "Level 3", "Level 4"], 120),
    "Outcome score": np.r_[rng.normal(40, 5, 120), rng.normal(46, 6, 120),
                           rng.normal(53, 8, 120), rng.normal(61, 10, 120)],
})


def box_levels():
    return boxplot(levels, x="Level", y="Outcome score")[0]


# scatter with fitted lines: the slope is steeper in Group A than in Group B
points = pd.DataFrame({"Input measure": rng.uniform(10, 60, 180)})
points["Group"] = rng.choice(["Group A", "Group B"], 180)
points["Outcome score"] = (
    20 + np.where(points["Group"] == "Group A", 0.9, 0.3) * points["Input measure"]
    + rng.normal(0, 6, 180)
)


def scatter_fit():
    return scatterplot(points, x="Input measure", y="Outcome score", group="Group")[0]


# two panels, one legend below both: the same three sites in two regions
starts_and_slopes = {
    "Region one": {"Site A": (35, 5.0), "Site B": (52.5, 0.0), "Site C": (45, -1.8)},
    "Region two": {"Site A": (30, 1.0), "Site B": (38.5, 4.0), "Site C": (62.5, -3.0)},
}
regions = {
    region: {site: start + slope * (periods - 1) + rng.normal(0, 0.25, len(periods))
             for site, (start, slope) in sites.items()}
    for region, sites in starts_and_slopes.items()
}


def panels_shared_legend():
    set_style("house")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, (region, sites) in zip(axes, regions.items()):
        for k, (site, values) in enumerate(sites.items()):
            ax.plot(periods, values, color="#000000", linestyle=BW_LINESTYLES[k],
                    marker=BW_MARKERS[k], markerfacecolor=BW_MARKER_FILLS[k],
                    markeredgecolor="#000000", label=site)
        ax.set_title(region, fontsize=12)
        ax.set_xlabel("Period")
        ax.tick_params(labelleft=True)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
    axes[0].set_ylabel("Index score")
    finish(fig)   # one boxed legend below both panels, limits padded
    return fig


# path diagram: every node is a box, every edge carries its coefficient
nodes = ["Background factor", "Second factor", "Intermediate step", "Later step", "Outcome"]
paths = [
    ("Background factor", "Intermediate step", "0.52"),
    ("Second factor", "Intermediate step", "0.18"),
    ("Intermediate step", "Later step", "0.47"),
    ("Later step", "Outcome", "0.39"),
    ("Background factor", "Outcome", "0.24"),
    ("Second factor", "Later step", "-0.21"),
]


def dag_paths():
    return dag(nodes, paths, highlight_nodes=["Background factor", "Outcome"],
               highlight_edges=[("Background factor", "Outcome")], figsize=(10, 5.5))


# co-occurrence network: Term A appears with a few terms far more often than with the rest
pair_counts = {"Term B": 30, "Term C": 22, "Term D": 15, "Term E": 9, "Term F": 5, "Term G": 3}
pairs = [["Term A", term] for term, count in pair_counts.items() for _ in range(count)]
documents = pd.Series(pairs + [["Term B", "Term C"]] * 8 + [["Term D", "Term E"]] * 4).to_frame("terms")


def cooccur_network():
    return cooccur(documents, "terms", top_n=1, max_neighbors=6, figsize=(8, 6))


FIGURES = {
    "bar_grouped": bar_grouped,
    "coef_intervals": coef_intervals,
    "trend_lines": trend_lines,
    "density_groups": density_groups,
    "box_levels": box_levels,
    "scatter_fit": scatter_fit,
    "panels_shared_legend": panels_shared_legend,
    "dag_paths": dag_paths,
    "cooccur_network": cooccur_network,
}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, make in FIGURES.items():
        fig = make()
        assert_clean(fig)
        save_figure(fig, OUT / name)
        plt.close("all")
        print(OUT / f"{name}.png")
