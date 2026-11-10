"""
layout.py: layout rules of the 'house' style, and the check that enforces them
------------------------------------------------------------------------------
The house style prevents overlap by construction and then measures the result:

- one legend for the whole figure, boxed and bold, below the panels
- long tick and legend labels broken over lines
- value labels spread evenly, never on top of each other
- axis limits padded until every marker, line end and label is inside the axes
- constrained layout, so text extents set the margins

``check_layout(fig)`` reads the bounding boxes of the rendered artists and
lists every overlap, clipped mark and off-canvas text. ``assert_clean(fig)``
raises on the first figure that has any.
"""

import textwrap

import numpy as np
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox

TICK_WRAP = 15       # tick labels longer than this are broken over lines
LEGEND_WRAP = 28     # legend labels longer than this are broken over lines
EDGE_LABEL_GID = "wtp-edge-label"

LABEL_BOX = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#000000",
                 linewidth=1.0, alpha=1.0)


# ══════════════════════════════════════════════════════════════════════════════
# I. LABELS
# ══════════════════════════════════════════════════════════════════════════════

def wrap_label(text, width=TICK_WRAP):
    """Break a label over lines at word boundaries once it passes `width` characters."""
    text = str(text)
    if len(text) <= width or "\n" in text:
        return text
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False,
                                   break_on_hyphens=False)) or text


def sentence_label(name):
    """
    Axis label from a column name: underscores become spaces and the first
    letter is capitalised; the rest is left as written.
    """
    text = str(name).replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else text


def wrap_tick_labels(ax, width=TICK_WRAP):
    """Break long categorical tick labels over lines. Numeric axes are left alone."""
    for axis in (ax.xaxis, ax.yaxis):
        labels = [t.get_text() for t in axis.get_ticklabels()]
        if not any(len(t) > width for t in labels):
            continue
        axis.set_ticks(axis.get_ticklocs())
        axis.set_ticklabels([wrap_label(t, width) for t in labels])


def integer_ticks(ax, axis="x"):
    """Whole-number ticks, for years and counts."""
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_locator(MaxNLocator(integer=True))


# ══════════════════════════════════════════════════════════════════════════════
# II. LEGEND
# ══════════════════════════════════════════════════════════════════════════════

def shared_legend(fig, handles=None, labels=None, title=None, ncol=None, max_cols=4):
    """
    One legend for the whole figure, centred below the panels.

    Opaque white box, dark border, bold title and labels. Entries are taken
    from every axes when not given; duplicates are dropped. Legends on the
    individual axes are removed. The figure is switched to constrained layout,
    which reserves the band the legend sits in.
    """
    if handles is None or labels is None:
        handles, labels, seen = [], [], set()
        for ax in fig.axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen:
                    seen.add(l)
                    handles.append(h)
                    labels.append(l)
    for ax in fig.axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    for old in list(fig.legends):
        old.remove()
    if not handles:
        return None

    use_constrained(fig)
    legend = fig.legend(
        handles, [wrap_label(l, LEGEND_WRAP) for l in labels],
        loc="outside lower center",
        ncol=ncol or min(len(labels), max_cols),
        title=title,
        frameon=True, fancybox=False, facecolor="white", framealpha=1.0, edgecolor="0.15",
        fontsize=10, title_fontsize=11,
        handlelength=2.5, handletextpad=0.8, columnspacing=2.0, borderpad=0.8,
    )
    legend.get_frame().set_linewidth(1.0)
    legend.get_title().set_fontweight("bold")
    for text in legend.get_texts():
        text.set_fontweight("bold")
    return legend


def use_constrained(fig):
    """Constrained layout with wide gaps between panels."""
    if fig.get_layout_engine() is None or type(fig.get_layout_engine()).__name__ != "ConstrainedLayoutEngine":
        fig.set_layout_engine("constrained")
    fig.get_layout_engine().set(w_pad=0.12, h_pad=0.10, wspace=0.10, hspace=0.08)


# ══════════════════════════════════════════════════════════════════════════════
# III. MEASURING THE RENDERED FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def _renderer(fig):
    fig.draw_without_rendering()
    return fig._get_renderer()


def _px(fig, points):
    return points * fig.dpi / 72.0


def _data_axes(fig):
    return [a for a in fig.axes if a.get_visible() and a.get_label() != "<colorbar>"]


def _is_data_line(ax, line):
    """Lines drawn in data coordinates (axhline/axvline reference lines are not)."""
    return line.get_transform() == ax.transData and line.get_visible()


def _text_box(text, renderer):
    """Window extent of a text, including its background box when it has one."""
    patch = text.get_bbox_patch()
    if patch is not None:
        return patch.get_window_extent(renderer)
    return text.get_window_extent(renderer)


def _point_marks(ax):
    """
    Every marker in the axes as (x, y, radius) in pixels: scatter points and
    line markers (trend points, estimates, outliers, network nodes).
    """
    fig = ax.figure
    out = []
    for coll in ax.collections:
        if not isinstance(coll, PathCollection) or not coll.get_visible():
            continue
        offsets = np.asarray(coll.get_offsets(), dtype=float)
        if offsets.size == 0:
            continue
        pts = coll.get_offset_transform().transform(offsets)
        sizes = np.asarray(coll.get_sizes(), dtype=float)
        if sizes.size == 0:
            continue
        sizes = np.resize(sizes, len(pts))
        lw = np.max(coll.get_linewidths()) if len(coll.get_linewidths()) else 0.0
        for (x, y), s in zip(pts, sizes):
            if np.isfinite(x) and np.isfinite(y):
                out.append((x, y, _px(fig, np.sqrt(s) / 2.0 + lw / 2.0)))
    for line in ax.lines:
        if not _is_data_line(ax, line) or line.get_marker() in (None, "", "None", " "):
            continue
        pts = ax.transData.transform(np.column_stack([line.get_xdata(orig=False),
                                                      line.get_ydata(orig=False)]).astype(float))
        r = _px(fig, line.get_markersize() / 2.0 + line.get_markeredgewidth() / 2.0)
        out.extend((x, y, r) for x, y in pts if np.isfinite(x) and np.isfinite(y))
    return out


def _line_points(ax, step=6.0):
    """
    Points sampled along every data line (about one per `step` pixels) as
    (x, y, half_width) in pixels: trend lines, fitted lines, interval bars,
    undirected network edges.
    """
    fig = ax.figure
    out = []

    def sample(pts, half):
        pts = pts[np.isfinite(pts).all(axis=1)]
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(int(np.hypot(*(b - a)) // step), 1)
            for t in np.linspace(0, 1, n + 1):
                out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), half))
        if len(pts) == 1:
            out.append((pts[0][0], pts[0][1], half))

    for line in ax.lines:
        if not _is_data_line(ax, line) or line.get_linestyle() in ("None", "", " "):
            continue
        xy = np.column_stack([line.get_xdata(orig=False), line.get_ydata(orig=False)]).astype(float)
        if len(xy):
            sample(ax.transData.transform(xy), _px(fig, line.get_linewidth() / 2.0))
    for coll in ax.collections:
        if isinstance(coll, LineCollection) and coll.get_visible():
            lw = np.max(coll.get_linewidths()) if len(coll.get_linewidths()) else 1.0
            for seg in coll.get_segments():
                if len(seg):
                    sample(coll.get_transform().transform(np.asarray(seg, dtype=float)),
                           _px(fig, lw / 2.0))
    return out


def _arrow_points(ax):
    """Points along every arrow (directed network edges), in pixels."""
    out = []
    for patch in ax.patches:
        if isinstance(patch, FancyArrowPatch) and patch.get_visible():
            path = patch.get_transform().transform_path(patch.get_path())
            for poly in path.to_polygons(closed_only=False):
                out.extend((x, y) for x, y in poly)
    return out


def _bars(ax, renderer):
    """Window extents of bars and boxes (rectangles and box-plot bodies)."""
    out = []
    for patch in ax.patches:
        if isinstance(patch, FancyArrowPatch) or not patch.get_visible() or patch is ax.patch:
            continue
        if isinstance(patch, Rectangle) and (patch.get_width() == 0 or patch.get_height() == 0):
            continue
        bb = patch.get_window_extent(renderer)
        if np.isfinite([bb.x0, bb.y0, bb.x1, bb.y1]).all() and bb.width > 0 and bb.height > 0:
            out.append((bb, _px(ax.figure, patch.get_linewidth() / 2.0)))
    return out


def _annotations(ax):
    """Texts placed inside the axes by a chart (value labels, node labels)."""
    return [t for t in ax.texts if t.get_visible() and t.get_text().strip()]


def _all_texts(fig, renderer):
    """Every visible, non-empty text in the figure as (kind, text, bbox)."""
    out = []

    def add(kind, text):
        if text is None or not text.get_visible() or not text.get_text().strip():
            return
        bb = _text_box(text, renderer)
        if bb.width > 0 and bb.height > 0:
            out.append((kind, text, bb))

    for ax in fig.axes:
        if not ax.get_visible():
            continue
        if ax.axison:
            for name, axis in (("xtick", ax.xaxis), ("ytick", ax.yaxis)):
                lo, hi = sorted(axis.get_view_interval())
                span = (hi - lo) or 1.0
                for tick in axis.get_major_ticks():
                    if lo - 1e-9 * span <= tick.get_loc() <= hi + 1e-9 * span:
                        add(name, tick.label1)
                        add(name, tick.label2)
                add("axis label", axis.get_label())
                add("axis label", axis.get_offset_text())
        for title in (ax.title, ax._left_title, ax._right_title):
            add("panel title", title)
        for text in _annotations(ax):
            add("label", text)
        if ax.get_legend() is not None:
            for text in [ax.get_legend().get_title()] + list(ax.get_legend().get_texts()):
                add("legend", text)
    for legend in fig.legends:
        for text in [legend.get_title()] + list(legend.get_texts()):
            add("legend", text)
    for text in fig.texts:
        add("figure text", text)
    if getattr(fig, "_suptitle", None) is not None:
        add("figure title", fig._suptitle)
    return out


def _legend_boxes(fig, renderer):
    out = [legend.get_window_extent(renderer) for legend in fig.legends if legend.get_visible()]
    for ax in fig.axes:
        if ax.get_legend() is not None and ax.get_legend().get_visible():
            out.append(ax.get_legend().get_window_extent(renderer))
    return out


def _overlap(a, b, tol=1.0):
    """True when two boxes share more than `tol` pixels in both directions."""
    return (min(a.x1, b.x1) - max(a.x0, b.x0) > tol) and (min(a.y1, b.y1) - max(a.y0, b.y0) > tol)


def _inside(bb, x, y, grow=0.0):
    return bb.x0 - grow < x < bb.x1 + grow and bb.y0 - grow < y < bb.y1 + grow


def _touches(bb, x, y, r):
    """True when a circle of radius r at (x, y) reaches into the box."""
    nx = min(max(x, bb.x0), bb.x1)
    ny = min(max(y, bb.y0), bb.y1)
    return np.hypot(x - nx, y - ny) < r


def _name(text):
    return repr(text.get_text().replace("\n", " ")[:40])


# ══════════════════════════════════════════════════════════════════════════════
# IV. PADDING AND SPREADING
# ══════════════════════════════════════════════════════════════════════════════

def pad_limits(ax, margin=4.0, sides="lrtb", rounds=4):
    """
    Widen the axis limits until every marker, line and in-axes label sits
    fully inside the axes, with `margin` points to spare.

    `sides` names the limits that may move (l, r, t, b), so a zero baseline
    can be kept. Log axes are left alone.
    """
    fig = ax.figure
    if ax.get_xscale() != "linear" or ax.get_yscale() != "linear":
        return
    pad = _px(fig, margin)
    for _ in range(rounds):
        renderer = _renderer(fig)
        box = ax.get_window_extent(renderer)
        ext = [(x - r, y - r, x + r, y + r) for x, y, r in _point_marks(ax)]
        ext += [(x - h, y - h, x + h, y + h) for x, y, h in _line_points(ax)]
        ext += [(x, y, x, y) for x, y in _arrow_points(ax)]
        for text in _annotations(ax):
            bb = _text_box(text, renderer)
            ext.append((bb.x0, bb.y0, bb.x1, bb.y1))
        if not ext or box.width <= 0 or box.height <= 0:
            return
        ext = np.asarray(ext, dtype=float)
        over = {
            "l": box.x0 + pad - ext[:, 0].min(),
            "b": box.y0 + pad - ext[:, 1].min(),
            "r": ext[:, 2].max() - (box.x1 - pad),
            "t": ext[:, 3].max() - (box.y1 - pad),
        }
        over = {k: v for k, v in over.items() if k in sides and v > 0.25}
        if not over:
            return
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        # pixels to data units; the 1.05 lets the loop settle in a round or two
        ux = (x1 - x0) / box.width * 1.05
        uy = (y1 - y0) / box.height * 1.05
        ax.set_xlim(x0 - over.get("l", 0) * ux, x1 + over.get("r", 0) * ux)
        ax.set_ylim(y0 - over.get("b", 0) * uy, y1 + over.get("t", 0) * uy)


def spread(positions, gap):
    """
    Move 1-D positions apart until neighbours are at least `gap` apart, keeping
    their order and staying as close to the originals as possible.
    """
    positions = np.asarray(positions, dtype=float)
    order = np.argsort(positions)
    out = positions[order].copy()
    for _ in range(200):
        moved = False
        for i in range(len(out) - 1):
            short = gap - (out[i + 1] - out[i])
            if short > 1e-9:
                out[i] -= short / 2.0
                out[i + 1] += short / 2.0
                moved = True
        if not moved:
            break
    result = np.empty_like(out)
    result[order] = out
    return result


def value_label(ax, x, y, text, dx=0.0, dy=0.0, ha="center", va="bottom", box=True, **kwargs):
    """
    A value label anchored at a data point and offset by (dx, dy) points, so
    it keeps its distance from the mark at any figure size.
    """
    style = dict(fontsize=10, fontweight="bold", color="black", zorder=6)
    style.update(kwargs)
    return ax.annotate(
        text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
        ha=ha, va=va, bbox=dict(LABEL_BOX) if box else None,
        annotation_clip=False, **style,
    )


def labels_fit(ax, labels, tol=1.0):
    """True when no two of the given labels overlap in the rendered figure."""
    renderer = _renderer(ax.figure)
    boxes = [_text_box(t, renderer) for t in labels]
    return not any(_overlap(a, b, tol) for i, a in enumerate(boxes) for b in boxes[i + 1:])


# ══════════════════════════════════════════════════════════════════════════════
# V. FINISHING A HOUSE FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def finish(fig, handles=None, labels=None, legend_title=None, legend=True, pad="lrtb",
           legend_cols=None):
    """
    Final pass of the house style: break long tick labels, place the shared
    legend below the panels, switch to constrained layout and pad the limits
    until nothing is clipped.

    Parameters
    ----------
    handles, labels : list, optional
        Legend entries; collected from the axes when omitted.
    legend : bool
        False draws no legend.
    pad : str or dict
        Sides of the limits that may be widened ('l', 'r', 't', 'b'); a dict
        maps each axes to its own sides.
    """
    axes = _data_axes(fig)
    use_constrained(fig)
    for ax in axes:
        wrap_tick_labels(ax)
    if legend:
        shared_legend(fig, handles, labels, title=legend_title, ncol=legend_cols)
    else:
        for ax in axes:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
    for ax in axes:
        sides = pad.get(ax, "lrtb") if isinstance(pad, dict) else pad
        if sides:
            pad_limits(ax, sides=sides)
    _renderer(fig)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# VI. THE CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_layout(fig, tol=1.5):
    """
    List every layout fault in a rendered figure. An empty list means clean.

    Checked, from the renderer's bounding boxes:

    - text over text (tick labels, axis labels, titles, value labels, legend)
    - legend over data (markers, lines, bars)
    - text outside the figure
    - data marks outside the axes area (markers, line ends, bars)
    - data marks under a label or a node box (markers, lines, arrows)
    - labels over bars

    Each fault is a dict with 'kind' and 'detail'.
    """
    renderer = _renderer(fig)
    canvas = Bbox.from_bounds(0, 0, *fig.get_size_inches() * fig.dpi)
    faults = []

    def fault(kind, detail):
        faults.append({"kind": kind, "detail": detail})

    texts = _all_texts(fig, renderer)
    for i, (kind_a, text_a, bb_a) in enumerate(texts):
        for kind_b, text_b, bb_b in texts[i + 1:]:
            if _overlap(bb_a, bb_b, tol):
                fault("text over text", f"{kind_a} {_name(text_a)} and {kind_b} {_name(text_b)}")
    for kind, text, bb in texts:
        if bb.x0 < -tol or bb.y0 < -tol or bb.x1 > canvas.x1 + tol or bb.y1 > canvas.y1 + tol:
            fault("text outside figure", f"{kind} {_name(text)}")

    legends = _legend_boxes(fig, renderer)
    for ax in _data_axes(fig):
        box = ax.get_window_extent(renderer)
        marks = _point_marks(ax)
        lines = _line_points(ax)
        arrows = _arrow_points(ax)
        bars = _bars(ax, renderer)
        notes = [(t, _text_box(t, renderer)) for t in _annotations(ax)]

        outside = sum(1 for x, y, r in marks
                      if x - r < box.x0 - tol or x + r > box.x1 + tol
                      or y - r < box.y0 - tol or y + r > box.y1 + tol)
        if outside:
            fault("mark outside axes", f"{outside} marker(s) cross the axes edge")
        # a line may lie along the axis (a density tail at zero) but not beyond it
        outside = sum(1 for x, y, h in lines
                      if x < box.x0 - tol or x > box.x1 + tol or y < box.y0 - tol or y > box.y1 + tol)
        if outside:
            fault("mark outside axes", f"{outside} line point(s) beyond the axes edge")
        outside = sum(1 for x, y in arrows if not _inside(box, x, y, grow=tol))
        if outside:
            fault("mark outside axes", f"{outside} arrow point(s) beyond the axes edge")
        for bb, half in bars:
            grow = tol + half
            if bb.x0 < box.x0 - grow or bb.x1 > box.x1 + grow or bb.y0 < box.y0 - grow or bb.y1 > box.y1 + grow:
                fault("mark outside axes", "a bar or box crosses the axes edge")
                break
        for text, bb in notes:
            if bb.x0 < box.x0 - tol or bb.x1 > box.x1 + tol or bb.y0 < box.y0 - tol or bb.y1 > box.y1 + tol:
                fault("label outside axes", _name(text))

        for lb in legends:
            hidden = sum(1 for x, y, r in marks if _touches(lb, x, y, r))
            hidden += sum(1 for x, y, h in lines if _inside(lb, x, y, grow=h))
            hidden += sum(1 for x, y in arrows if _inside(lb, x, y))
            hidden += sum(1 for bb, _ in bars if _overlap(lb, bb, tol))
            if hidden:
                fault("legend over data", f"{hidden} mark(s) under the legend")

        for text, bb in notes:
            hidden = sum(1 for x, y, r in marks if _touches(bb, x, y, r - tol))
            if hidden:
                fault("mark under label", f"{hidden} marker(s) under {_name(text)}")
            hidden = sum(1 for x, y, h in lines if _inside(bb, x, y, grow=-tol))
            if hidden:
                fault("mark under label", f"line under {_name(text)}")
            if text.get_gid() != EDGE_LABEL_GID:
                hidden = sum(1 for x, y in arrows if _inside(bb, x, y, grow=-tol))
                if hidden:
                    fault("mark under label", f"edge under {_name(text)}")
            if any(_overlap(bb, bar, tol) for bar, _ in bars):
                fault("label over bar", _name(text))
    return faults


def assert_clean(fig, tol=1.5):
    """Raise AssertionError listing the layout faults of `fig`, if it has any."""
    faults = check_layout(fig, tol=tol)
    if faults:
        lines = "\n".join(f"  {f['kind']}: {f['detail']}" for f in faults)
        raise AssertionError(f"{len(faults)} layout fault(s):\n{lines}")
    return fig
