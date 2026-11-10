"""
Layout tests: the overlap and clipping check itself, and the check run on
figures drawn in the default 'house' style.

check_layout reads the bounding boxes of every text, legend and data mark from
the renderer. A clean figure returns an empty list.
"""

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from whatsthepoint import visuals as V
from whatsthepoint.utils.layout import (
    assert_clean, check_layout, finish, spread, value_label, wrap_label,
)
from whatsthepoint.utils.style import AVAILABLE_STYLES, set_style

from test_smoke import BARS, COEFS, DF, GROUPED, TERMS, TIME, has

SAMPLES = Path(__file__).resolve().parent.parent / "docs" / "sim_make_samples.py"
network = pytest.mark.skipif(not has("networkx"), reason="needs networkx")

DAG_NODES = ["First cause", "Second cause", "Middle step", "Later step", "Result"]
DAG_EDGES = [("First cause", "Middle step", "+"), ("Second cause", "Middle step", "+"),
             ("Middle step", "Later step", "+"), ("Later step", "Result", "+"),
             ("First cause", "Result", "+"), ("Second cause", "Later step", "-")]

HOUSE_FIGURES = {
    "bar vertical": lambda: V.bar.bar(BARS, "category", "value")[0],
    "bar horizontal": lambda: V.bar.bar(BARS, "category", "value", orientation="horizontal")[0],
    "bar grouped": lambda: V.bar.bar(GROUPED, "wave", "value", orientation="grouped")[0],
    "bar stacked": lambda: V.bar.bar(GROUPED, "wave", "value", orientation="stacked")[0],
    "coef": lambda: V.coef.coef(COEFS, p_col="p.value")[0],
    "trend": lambda: V.trend.trend(TIME, "date", "value", group="category")[0],
    "scatter": lambda: V.scatter.scatterplot(DF, "x", "y", group="sex")[0],
    "kde": lambda: V.density.kde(DF, "y", group="sex")[0],
    "boxplot": lambda: V.boxplot.boxplot(DF, "group", "y")[0],
    "dag": pytest.param(lambda: V.dag.dag(DAG_NODES, DAG_EDGES, highlight_nodes=["Result"]),
                        marks=network),
    "dag chain": pytest.param(lambda: V.dag.dag(["X", "M", "Y"], [("X", "M"), ("M", "Y"), ("X", "Y")]),
                              marks=network),
    "cooccur": pytest.param(lambda: V.cooccur.cooccur(TERMS, "terms", top_n=2), marks=network),
}


def _cases(figures):
    for name, make in figures.items():
        if isinstance(make, type(pytest.param(None))):
            yield pytest.param(*make.values, marks=make.marks, id=name)
        else:
            yield pytest.param(make, id=name)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def kinds(fig):
    return {fault["kind"] for fault in check_layout(fig)}


# ── the check finds each fault it is meant to find ────────────────────────────

def plain_axes():
    set_style("house")
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.plot([0, 1, 2], [0, 1, 2], marker="o", color="black")
    return fig, ax


def test_clean_figure_has_no_faults():
    fig, ax = plain_axes()
    finish(fig, legend=False)
    assert check_layout(fig) == []
    assert_clean(fig)


def test_finds_text_over_text():
    fig, ax = plain_axes()
    ax.text(1.0, 0.5, "first label")
    ax.text(1.02, 0.5, "second label")
    assert "text over text" in kinds(fig)


def test_finds_legend_over_data():
    fig, ax = plain_axes()
    ax.legend(["series"], loc="center")
    assert "legend over data" in kinds(fig)


def test_finds_text_outside_figure():
    fig, ax = plain_axes()
    fig.text(1.2, 0.5, "off the canvas")
    assert "text outside figure" in kinds(fig)


def test_finds_marker_cut_by_axes_edge():
    fig, ax = plain_axes()
    ax.set_xlim(0, 2)          # the first and last markers sit on the edge
    ax.set_ylim(0, 2)
    assert "mark outside axes" in kinds(fig)


def test_finds_marker_under_label():
    fig, ax = plain_axes()
    ax.text(1, 1, "label", ha="center", va="center", bbox=dict(facecolor="white"))
    assert "mark under label" in kinds(fig)


def test_finds_label_over_bar():
    set_style("house")
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.bar([0, 1], [3, 5], color="white", edgecolor="black")
    ax.text(1, 4.9, "5", ha="center", va="center")
    assert "label over bar" in kinds(fig)


def test_assert_clean_raises_with_the_faults_listed():
    fig, ax = plain_axes()
    ax.legend(["series"], loc="center")
    with pytest.raises(AssertionError, match="legend over data"):
        assert_clean(fig)


def test_finish_pads_limits_until_marks_are_inside():
    fig, ax = plain_axes()
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    value_label(ax, 2, 2, "2.0", dx=12, ha="left", va="center")
    finish(fig, legend=False)
    assert check_layout(fig) == []


def test_finish_puts_one_boxed_bold_legend_below_the_panels():
    set_style("house")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=100)
    for ax in axes:
        ax.plot([0, 1], [0, 1], color="black", label="first series")
        ax.plot([0, 1], [1, 0], color="black", linestyle="--", label="second series")
        ax.legend()
    finish(fig)
    assert all(ax.get_legend() is None for ax in axes)
    assert len(fig.legends) == 1
    legend = fig.legends[0]
    assert [t.get_text() for t in legend.get_texts()] == ["first series", "second series"]
    assert legend.get_frame_on() and legend.get_frame().get_alpha() == 1.0
    assert all(t.get_fontweight() == "bold" for t in legend.get_texts())
    renderer = fig.canvas.get_renderer()
    top_of_legend = legend.get_window_extent(renderer).y1
    assert all(top_of_legend < ax.get_tightbbox(renderer).y0 for ax in axes)
    assert check_layout(fig) == []


def test_wrap_label_breaks_long_labels_at_words():
    assert wrap_label("Short") == "Short"
    assert wrap_label("A much longer category label") == "A much longer\ncategory label"
    assert all(len(line) <= 15 for line in wrap_label("A much longer category label").split("\n"))


def test_spread_keeps_order_and_minimum_gap():
    out = spread([10.0, 10.5, 30.0, 11.0], gap=5)
    ordered = sorted(out)
    assert all(b - a >= 5 - 1e-6 for a, b in zip(ordered, ordered[1:]))
    assert out[0] < out[1] < out[3] < out[2]
    assert out[2] == 30.0


# ── figures in the house style are clean ──────────────────────────────────────

@pytest.mark.parametrize("make", list(_cases(HOUSE_FIGURES)))
def test_house_figure_is_clean(make):
    assert_clean(make())


def test_house_is_the_default_and_other_modes_still_draw():
    from whatsthepoint.utils import style
    assert AVAILABLE_STYLES[0] == "house"
    V.bar.bar(GROUPED, "wave", "value", orientation="grouped")
    assert style.ACTIVE_STYLE == "house"
    for mode in AVAILABLE_STYLES:
        fig, ax = V.bar.bar(GROUPED, "wave", "value", orientation="grouped", style_mode=mode)
        assert len(ax.patches) == 6
        plt.close(fig)


def test_house_grouped_bars_are_black_and_white_with_value_labels():
    fig, ax = V.bar.bar(GROUPED, "wave", "value", orientation="grouped")
    faces = {tuple(round(c, 2) for c in p.get_facecolor()[:3]) for p in ax.patches}
    assert faces == {(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)}
    assert sorted(t.get_text() for t in ax.texts) == ["35", "38", "40", "41", "45", "50"]
    assert ax.get_ylim()[0] == 0


@network
def test_dag_nodes_are_their_label_boxes_and_edges_carry_labels():
    fig = V.dag.dag(DAG_NODES, DAG_EDGES)
    ax = fig.axes[0]
    node_boxes = [t for t in ax.texts if t.get_text().replace("\n", " ") in DAG_NODES]
    assert len(node_boxes) == len(DAG_NODES)
    assert all(t.get_bbox_patch() is not None for t in node_boxes)
    assert len(ax.collections) == 0            # no separate node markers to hide
    assert len([t for t in ax.texts if t not in node_boxes]) == len(DAG_EDGES)
    assert_clean(fig)


def test_house_font_falls_back_to_the_one_matplotlib_ships():
    from matplotlib import font_manager
    set_style("house")
    assert plt.rcParams["font.sans-serif"][0] == "DejaVu Sans"
    family = font_manager.FontProperties(family=plt.rcParams["font.sans-serif"])
    found = font_manager.findfont(family, fallback_to_default=False)
    assert Path(found).name.startswith("DejaVuSans")
    assert Path(found).is_relative_to(Path(matplotlib.get_data_path()))


def test_resolve_data_root_uses_argument_then_environment(tmp_path, monkeypatch):
    from whatsthepoint.data import resolve_data_root
    (tmp_path / "survey_files" / "inner").mkdir(parents=True)
    (tmp_path / "other_store" / "inner").mkdir(parents=True)
    assert resolve_data_root(tmp_path / "survey_files" / "inner", ["survey_files"]).name == "survey_files"
    monkeypatch.setenv("WHATSTHEPOINT_DATA", "other_store")
    assert resolve_data_root(tmp_path / "other_store" / "inner").name == "other_store"


# ── every committed sample figure is clean ────────────────────────────────────

def _sample_figures():
    if not SAMPLES.exists():
        return {}
    spec = importlib.util.spec_from_file_location("sim_make_samples", SAMPLES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FIGURES


@pytest.mark.skipif(not SAMPLES.exists(), reason="docs/sim_make_samples.py not present")
def test_every_sample_figure_is_clean():
    figures = _sample_figures()
    assert figures
    for name, make in figures.items():
        try:
            assert_clean(make())
        except AssertionError as err:
            raise AssertionError(f"{name}: {err}") from None
        plt.close("all")
