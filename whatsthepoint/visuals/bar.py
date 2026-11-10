"""
bar.py: whatsthepoint Visualization Module
--------------------------------------------
Flexible categorical comparisons:
- vertical, horizontal, or stacked bars
- optional highlight bar
- subplot support for comparing multiple distributions
- consistent whatsthepoint styling
- Plotly interactive counterpart
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import make_interp_spline
from ..utils.style import (
    set_style,
    generate_semantic_palette,
    apply_titles,
    get_data_element_kwargs,
    format_tick_labels,
    get_hatches,
    smart_format,
    zero_baseline,
    legend_below,
    is_house,
)
from ..utils.layout import finish, value_label, labels_fit, sentence_label


# ══════════════════════════════════════════════════════════════════════════════
# STATIC VERSION
# ══════════════════════════════════════════════════════════════════════════════

def bar(
    df,
    x,
    y,
    title=None,
    subtitle=None,
    palette=None,
    n=None,
    style_mode="house",
    orientation="vertical",        # 'vertical', 'horizontal', 'stacked', 'grouped'
    highlight=None,                # highlight label
    highlight_color="#D62828",
    trace_line=False,
    trace_arrow=True,
    sort="none",                   # 'none', 'asc', or 'desc'
    group_spacing=None,            # e.g. [(0,2), (3,5)] or int for split index
    figsize=(10, 6),
    legend_title=None,
):
    """
    whatsthepoint bar plot with multiple orientations and optional features.

    Parameters
    ----------
    df : pd.DataFrame
        Data for plotting
    x : str
        Column name for x-axis categories
    y : str
        Column name for y-axis values (not used for stacked/grouped)
    title : str, optional
        Plot title (None = no title)
    subtitle : str, optional
        Plot subtitle
    palette : dict, optional
        Custom color mapping (auto-generated if None)
    n : int, optional
        Sample size annotation
    style_mode : str, default "house"
        whatsthepoint style mode
    orientation : str, default 'vertical'
        Bar orientation:
        - 'vertical': Standard vertical bars
        - 'horizontal': Horizontal bars
        - 'stacked': Stacked bars (columns as stack segments)
        - 'grouped': Grouped bars side-by-side (columns as groups)
    highlight : str, optional
        Category to highlight with special color
    highlight_color : str, default '#D62828'
        Color for highlighted category
    trace_line : bool, default False
        Draw curved line connecting bar tops (vertical only)
    trace_arrow : bool, default True
        Add arrowhead to trace line
    sort : str, default 'none'
        Sort bars: 'none', 'asc', or 'desc'
    group_spacing : int, list, or tuple, optional
        Add gaps between bar groups
    figsize : tuple, default (10, 6)
        Figure size in inches.
    legend_title : str, optional
        Title for the legend of stacked and grouped bars. None by default.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes

    Examples
    --------
    Simple vertical bars:
    >>> bar(df, x='category', y='value')

    Stacked bars (distribution over time):
    >>> bar(df, x='wave', y='wave', orientation='stacked')

    Grouped bars (side-by-side comparison):
    >>> bar(df, x='wave', y='wave', orientation='grouped')

    Notes
    -----
    For stacked and grouped orientations:
    - Each column (except x and y) becomes a segment/group
    - Uses discrete high-contrast color palette for up to 4 groups
    - Legend appears on the right side
    - X-axis labels auto-rotate at 45° if >6 labels or long text
    """

    # ─────────────────────────────────────────────
    # Sort and group spacing
    # ─────────────────────────────────────────────
    df = df.copy()
    if sort == "asc":
        df = df.sort_values(y, ascending=True)
    elif sort == "desc":
        df = df.sort_values(y, ascending=False)

    # Add gaps between groups if requested
    if isinstance(group_spacing, int):
        split_points = [group_spacing]
    elif isinstance(group_spacing, (list, tuple)):
        split_points = [g[1] for g in group_spacing]
    else:
        split_points = []

    # Apply pseudo-gap by inserting NaN rows
    if split_points:
        dfs = []
        last = 0
        for sp in split_points:
            dfs.append(df.iloc[last:sp])
            dfs.append(
                {x: f"", y: np.nan}
            )  # add blank separator
            last = sp
        dfs.append(df.iloc[last:])
        df = (
            pd.concat([pd.DataFrame(d) if not isinstance(d, dict) else pd.DataFrame([d]) for d in dfs])
            .reset_index(drop=True)
        )

    # ─────────────────────────────────────────────
    # Styling setup
    # ─────────────────────────────────────────────
    set_style(style_mode)
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    if palette is None:
        groups = {"positive": [v for v in df[x].dropna().unique().tolist() if v != ""]}
        palette = generate_semantic_palette(groups, mode=style_mode)

    colors = [
        "white" if v == "" else (
            highlight_color if (highlight and v == highlight) else palette.get(v, matplotlib.colormaps["viridis"](0.6))
        )
        for v in df[x]
    ]
    _cats = [v for v in df[x].dropna().unique().tolist() if v != ""]
    _cat_hatch = dict(zip(_cats, get_hatches(len(_cats), style_mode)))
    hatches = [_cat_hatch.get(v, "") for v in df[x]]
    kwargs = get_data_element_kwargs()
    house = is_house(style_mode)
    name = sentence_label if house else (lambda s: s.title())
    value_labels = []

    # ─────────────────────────────────────────────
    # Main plotting logic
    # ─────────────────────────────────────────────
    if orientation == "horizontal":
        ax.barh(df[x], df[y], color=colors, hatch=hatches, **kwargs)
        ax.set_xlabel(name(y), fontsize=12, weight="bold", color="black")
        ax.set_ylabel("")
        for i, val in enumerate(df[y]):
            if house and not np.isnan(val):
                value_labels.append(value_label(ax, val, i, smart_format(val), dx=8, ha="left", va="center"))
            elif not np.isnan(val):
                ax.text(val + (df[y].max() * 0.015), i, smart_format(val), va="center", fontsize=9, color="black", weight="bold",
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#333333', linewidth=1.5, alpha=0.95))

    elif orientation == "stacked":
        cols = [c for c in df.columns if c not in [x, y]]
        # Filter out columns with no meaningful data (all NaN, all zeros, or negligible values)
        def has_meaningful_data(series):
            vals = series.dropna()
            if vals.empty:
                return False
            # Check if all values are effectively zero
            if (vals.abs() < 1e-10).all():
                return False
            # Check if the total contribution is meaningful (> 0.01% of any bar)
            if vals.sum() < 1e-10:
                return False
            return True

        cols = [c for c in cols if has_meaningful_data(df[c])]

        # Generate palette for stacked columns (not x-axis values!)
        stack_palette = generate_semantic_palette({"positive": cols}, mode=style_mode)

        bottom = np.zeros(len(df))
        stack_hatches = get_hatches(len(cols), style_mode)
        for c, h in zip(cols, stack_hatches):
            vals = df[c].values
            ax.bar(df[x], vals, bottom=bottom, label=c, color=stack_palette.get(c, matplotlib.colormaps["viridis"](0.6)), hatch=h, **kwargs)
            bottom += vals
        ax.set_ylabel("Total", fontsize=12, weight="bold", color="black")
        ax.set_xlabel(name(x), fontsize=12, weight="bold", color="black")

    elif orientation == "grouped":
        cols = [c for c in df.columns if c not in [x, y]]
        # Filter out columns with no meaningful data
        def has_meaningful_data(series):
            vals = series.dropna()
            if vals.empty:
                return False
            if (vals.abs() < 1e-10).all():
                return False
            if vals.sum() < 1e-10:
                return False
            return True

        cols = [c for c in cols if has_meaningful_data(df[c])]
        n_groups = len(cols)

        # Generate palette for grouped columns
        group_palette = generate_semantic_palette({"positive": cols}, mode=style_mode)

        # Calculate bar positions
        x_positions = np.arange(len(df))
        bar_width = 0.8 / n_groups  # Total width of 0.8 divided by number of groups

        # Plot grouped bars
        group_hatches = get_hatches(n_groups, style_mode)
        for i, col in enumerate(cols):
            # Calculate offset for this group
            offset = (i - n_groups/2 + 0.5) * bar_width
            vals = df[col].values
            ax.bar(x_positions + offset, vals, bar_width,
                   label=col,
                   color=group_palette.get(col, matplotlib.colormaps["viridis"](0.6)),
                   hatch=group_hatches[i],
                   **kwargs)
            if house:
                for xp, val in zip(x_positions + offset, vals):
                    if not np.isnan(val):
                        value_labels.append(value_label(ax, xp, val, smart_format(val), dy=6))

        # Set x-axis labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(df[x])

        # Legend

        ax.set_ylabel(name(y) if y else "Value", fontsize=12, weight="bold", color="black")
        ax.set_xlabel(name(x), fontsize=12, weight="bold", color="black")

    else:  # vertical
        ax.bar(df[x], df[y], color=colors, hatch=hatches, **kwargs)
        ax.set_xlabel(name(x), fontsize=12, weight="bold", color="black")
        ax.set_ylabel(name(y), fontsize=12, weight="bold", color="black")

        for i, val in enumerate(df[y]):
            if house and not np.isnan(val):
                value_labels.append(value_label(ax, i, val, smart_format(val), dy=6))
            elif not np.isnan(val):
                ax.text(i, val + (df[y].max() * 0.03), smart_format(val), ha="center", fontsize=9, color="black", weight="bold",
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#333333', linewidth=1.5, alpha=0.95))

        # ─────────────────────────────────────────────
        # Optional trace line and arrowhead
        # ─────────────────────────────────────────────
        if trace_line:
            # Smooth curve between bar tops
            valid_mask = ~df[y].isna()
            x_idx = np.arange(len(df))[valid_mask]
            y_vals = df[y][valid_mask].values

            # cubic spline smoothing
            spl = make_interp_spline(x_idx, y_vals, k=2)
            xs = np.linspace(x_idx.min(), x_idx.max(), 300)
            ys = spl(xs)

            # draw the curve + dots
            ax.plot(xs, ys, color="grey", lw=1.3, alpha=0.85, zorder=3)
            ax.scatter(x_idx[:-1], y_vals[:-1], color="grey", s=15, zorder=4)

            if trace_arrow:
                # ---- Arrow at end of curve with better positioning ----
                x_end, y_end = xs[-1], ys[-1]
                # Use larger step back for clearer direction
                step_back = min(10, len(xs) // 10)
                x_prev, y_prev = xs[-step_back], ys[-step_back]

                ax.annotate(
                    "",
                    xy=(x_end, y_end),
                    xytext=(x_prev, y_prev),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="grey",
                        lw=2.0,
                        alpha=0.85,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                    zorder=6,
                )


    # ─────────────────────────────────────────────
    # Y-axis padding to avoid visual exaggeration
    # ─────────────────────────────────────────────
    if orientation != "horizontal":
        # Check if ACTUAL DATA appears to be percentage (0-100) or proportion (0-1)
        # Look at the data values, not the autoscaled axis limits
        if orientation == "stacked":
            # For stacked bars, check the max of the stacked totals
            data_max = df[[c for c in df.columns if c not in [x, y]]].sum(axis=1).max()
            data_min = 0  # Stacked bars start at 0
        elif orientation == "grouped":
            # For grouped bars, check max across all group columns
            data_max = df[[c for c in df.columns if c not in [x, y]]].max().max()
            data_min = df[[c for c in df.columns if c not in [x, y]]].min().min()
        else:
            # For vertical bars, check y column
            data_max = df[y].max()
            data_min = df[y].min()

        # Use small tolerance for floating point comparison
        is_percentage = (data_max <= 100.0 + 1e-9 and data_min >= -1e-9)
        is_proportion = (data_max <= 1.0 + 1e-9 and data_min >= -1e-9)

        # Calculate padding based on ACTUAL DATA range, not autoscaled limits
        data_range = data_max - data_min

        # Value axis starts at 0 unless the data go negative
        padding = data_range * 0.20 if data_range > 0 else abs(data_max) * 0.20 or 1.0
        new_min = 0 if data_min >= 0 else data_min - padding
        new_max = data_max + padding
        if is_percentage and not is_proportion:
            new_max = min(100.0, new_max) if data_max < 100.0 else data_max
        elif is_proportion:
            new_max = min(1.0, new_max) if data_max < 1.0 else data_max
        if new_max <= new_min:
            new_max = new_min + 1.0

        ax.set_ylim(new_min, new_max)
    else:
        zero_baseline(ax, df[y], axis="x")

    # ─────────────────────────────────────────────
    # Styling and finishing touches
    # ─────────────────────────────────────────────
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # house: no note line in the image, so the sample size is not drawn
    apply_titles(fig, title, subtitle, n=None if house else n)

    # Automatic x-axis rotation for vertical/stacked bars with many or long labels
    rotation_angle = 0
    if orientation != "horizontal":
        x_labels = [str(label.get_text()) for label in ax.get_xticklabels()]
        n_labels = len(x_labels)
        avg_label_len = np.mean([len(label) for label in x_labels]) if x_labels else 0

        # Rotate if: more than 6 labels OR average label length > 8 characters
        # house breaks long labels over lines instead of turning them
        if (n_labels > 8) if house else (n_labels > 6 or avg_label_len > 8):
            rotation_angle = 45

    # Format tick labels: bold and conditionally angled
    format_tick_labels(ax, rotation_x=rotation_angle)

    if house:
        # one boxed legend below the axes; limits widened until every value
        # label is inside. Labels that cannot fit side by side lose their
        # boxes first and are dropped only if they still collide.
        has_legend = orientation in ["stacked", "grouped"]
        side = "r" if orientation == "horizontal" else "t"
        finish(fig, legend=has_legend, legend_title=legend_title, pad=side)
        if value_labels and not labels_fit(ax, value_labels):
            for t in value_labels:
                t.set_bbox(None)
            if not labels_fit(ax, value_labels):
                for t in value_labels:
                    t.remove()
            finish(fig, legend=has_legend, legend_title=legend_title, pad=side)
        return fig, ax

    # Stacked and grouped bars get one legend centred below the axes
    if orientation in ["stacked", "grouped"]:
        legend_below(ax, title=legend_title)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# SUBPLOTS VERSION FOR HORIZONTAL BAR CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def bar_subplots(
    df,
    x,
    y,
    facet_col=None,
    facet_row=None,
    title=None,
    subtitle=None,
    palette=None,
    style_mode="house",
    orientation="horizontal",
    highlight=None,
    highlight_color="#D62828",
    figsize=None,
    n=None,
):
    """
    Create subplots of bar charts to compare multiple distributions.

    Parameters
    ----------
    df : pd.DataFrame
        Data for plotting
    x : str
        Category variable name
    y : str or list
        Value variable(s). If list, creates one subplot per variable.
        If single string with facet_col/facet_row, creates subplots by facet.
    facet_col : str, optional
        Column to facet by (creates columns of subplots)
    facet_row : str, optional
        Row to facet by (creates rows of subplots)
    title : str, optional
        Overall title
    subtitle : str, optional
        Overall subtitle
    palette : dict, optional
        Color mapping
    style_mode : str
        whatsthepoint style mode
    orientation : str
        'horizontal' or 'vertical'
    highlight : str, optional
        Category to highlight
    highlight_color : str
        Color for highlighted category
    figsize : tuple, optional
        Figure size (width, height)
    n : int, optional
        Sample size annotation

    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    set_style(style_mode)

    # Determine subplot layout
    if isinstance(y, list):
        # Multiple y variables
        n_plots = len(y)
        n_cols = 2 if n_plots > 1 else 1
        n_rows = int(np.ceil(n_plots / n_cols))
        plot_type = "multi_y"
        y_vars = y
    elif facet_col is not None or facet_row is not None:
        # Faceting by category
        if facet_col and facet_row:
            col_vals = df[facet_col].unique()
            row_vals = df[facet_row].unique()
            n_cols = len(col_vals)
            n_rows = len(row_vals)
        elif facet_col:
            col_vals = df[facet_col].unique()
            n_cols = len(col_vals)
            n_rows = 1
            row_vals = [None]
        else:  # facet_row
            row_vals = df[facet_row].unique()
            n_rows = len(row_vals)
            n_cols = 1
            col_vals = [None]
        plot_type = "facet"
        y_vars = [y]
    else:
        raise ValueError("Must specify either multiple y variables or facet_col/facet_row")

    # Create figure
    if figsize is None:
        figsize = (6 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=130)
    fig.set_facecolor("white")

    # Ensure axes is always 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Generate palette if not provided
    if palette is None:
        groups = {"positive": df[x].unique().tolist()}
        palette = generate_semantic_palette(groups, mode=style_mode)

    _sub_cats = list(palette.keys())
    _sub_hatch = dict(zip(_sub_cats, get_hatches(len(_sub_cats), style_mode)))
    kwargs = get_data_element_kwargs()

    # Create subplots
    if plot_type == "multi_y":
        for idx, y_var in enumerate(y_vars):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            ax.set_facecolor("white")

            # Prepare data
            plot_df = df[[x, y_var]].copy()

            colors = [
                highlight_color if (highlight and v == highlight) else palette.get(v, matplotlib.colormaps["viridis"](0.6))
                for v in plot_df[x]
            ]

            if orientation == "horizontal":
                ax.barh(plot_df[x], plot_df[y_var], color=colors, hatch=[_sub_hatch.get(v, "") for v in plot_df[x]], **kwargs)
                ax.set_xlabel(y_var.replace("_", " ").title(), fontsize=11, weight="bold", color="black")
                ax.set_ylabel("")
                # Annotations
                for i, val in enumerate(plot_df[y_var]):
                    if not np.isnan(val):
                        ax.text(val + (plot_df[y_var].max() * 0.015), i, smart_format(val),
                               va="center", fontsize=9, color="black", weight="bold",
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#333333', linewidth=1.5, alpha=0.95))
            else:
                ax.bar(plot_df[x], plot_df[y_var], color=colors, hatch=[_sub_hatch.get(v, "") for v in plot_df[x]], **kwargs)
                ax.set_ylabel(y_var.replace("_", " ").title(), fontsize=11, weight="bold", color="black")
                ax.set_xlabel("")
                # Annotations
                for i, val in enumerate(plot_df[y_var]):
                    if not np.isnan(val):
                        ax.text(i, val + (plot_df[y_var].max() * 0.03), smart_format(val),
                               ha="center", fontsize=9, color="black", weight="bold",
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#333333', linewidth=1.5, alpha=0.95))

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Subplot title
            ax.set_title(y_var.replace("_", " ").title(), fontsize=12, weight="bold", pad=10)

    else:  # facet
        for row_idx, row_val in enumerate(row_vals):
            for col_idx, col_val in enumerate(col_vals):
                ax = axes[row_idx, col_idx]
                ax.set_facecolor("white")

                # Filter data
                if facet_col and facet_row:
                    plot_df = df[(df[facet_col] == col_val) & (df[facet_row] == row_val)].copy()
                    subplot_title = f"{col_val} | {row_val}"
                elif facet_col:
                    plot_df = df[df[facet_col] == col_val].copy()
                    subplot_title = str(col_val)
                else:
                    plot_df = df[df[facet_row] == row_val].copy()
                    subplot_title = str(row_val)

                colors = [
                    highlight_color if (highlight and v == highlight) else palette.get(v, matplotlib.colormaps["viridis"](0.6))
                    for v in plot_df[x]
                ]

                if orientation == "horizontal":
                    ax.barh(plot_df[x], plot_df[y], color=colors, hatch=[_sub_hatch.get(v, "") for v in plot_df[x]], **kwargs)
                    if row_idx == n_rows - 1:
                        ax.set_xlabel(y.replace("_", " ").title(), fontsize=11, weight="bold", color="black")
                    else:
                        ax.set_xlabel("")
                    if col_idx == 0:
                        ax.set_ylabel("")
                    # Annotations
                    for i, val in enumerate(plot_df[y]):
                        if not np.isnan(val):
                            ax.text(val + (plot_df[y].max() * 0.015), i, smart_format(val),
                                   va="center", fontsize=9, color="black", weight="bold",
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#333333', linewidth=1.5, alpha=0.95))
                else:
                    ax.bar(plot_df[x], plot_df[y], color=colors, hatch=[_sub_hatch.get(v, "") for v in plot_df[x]], **kwargs)
                    if col_idx == 0:
                        ax.set_ylabel(y.replace("_", " ").title(), fontsize=11, weight="bold", color="black")
                    else:
                        ax.set_ylabel("")
                    if row_idx == n_rows - 1:
                        ax.set_xlabel("")
                    # Annotations
                    for i, val in enumerate(plot_df[y]):
                        if not np.isnan(val):
                            ax.text(i, val + (plot_df[y].max() * 0.03), smart_format(val),
                                   ha="center", fontsize=9, color="black", weight="bold",
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#333333', linewidth=1.5, alpha=0.95))

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                # Subplot title
                ax.set_title(subplot_title, fontsize=12, weight="bold", pad=10)

    # Overall title
    apply_titles(fig, title, subtitle, n=n)
    fig.tight_layout(rect=(0, 0, 1, 0.9 if subtitle else 0.94))
    return fig, axes


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE VERSION
# ══════════════════════════════════════════════════════════════════════════════

def bar_interactive(
    df,
    x,
    y,
    title=None,
    subtitle=None,
    style_mode="house",
    orientation="vertical",
    highlight=None,
    highlight_color="#D62828",
    trace_line=False,
    color_mode="categorical",  # "categorical" or "continuous"
    group_col=None,
    show_values=True,
):
    """
    whatsthepoint interactive bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        Data for plotting.
    x, y : str
        Variable names for categories and values.
    title, subtitle : str, optional
        Title and subtitle text.
    style_mode : str
        whatsthepoint visual mode ('viridis', 'reviewer3', etc.).
    orientation : str
        'vertical' or 'horizontal'.
    highlight : str, optional
        Category name to highlight.
    trace_line : bool, optional
        Draw connecting line across bar tops.
    color_mode : str
        "categorical" (distinct palette) or "continuous" (value gradient).
    group_col : str, optional
        Optional grouping column for coloring.
    show_values : bool, optional
        Annotate bars with numeric labels.
    """
    set_style(style_mode)

    # --- Color logic ---
    if color_mode == "continuous":
        cmap = matplotlib.colormaps["viridis"]
        norm_vals = (df[y] - df[y].min()) / (df[y].max() - df[y].min() + 1e-9)
        colors = [f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a:.2f})"
                  for r, g, b, a in cmap(norm_vals)]
    elif group_col and group_col in df.columns:
        groups = {"positive": df[group_col].unique().tolist()}
        palette = generate_semantic_palette(groups, mode=style_mode)
        colors = [palette.get(v, matplotlib.colormaps["viridis"](0.6)) for v in df[group_col]]
        # Convert to rgba
        colors = [f"rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},{c[3]:.2f})"
                 if isinstance(c, tuple) else c for c in colors]
    else:
        if highlight:
            groups = {"positive": df[x].unique().tolist()}
            palette = generate_semantic_palette(groups, mode=style_mode)
            colors = []
            for v in df[x]:
                if v == highlight:
                    colors.append(highlight_color)
                else:
                    c = palette.get(v, matplotlib.colormaps["viridis"](0.6))
                    colors.append(f"rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},{c[3]:.2f})")
        else:
            groups = {"positive": df[x].unique().tolist()}
            palette = generate_semantic_palette(groups, mode=style_mode)
            colors = [palette.get(v, matplotlib.colormaps["viridis"](0.6)) for v in df[x]]
            colors = [f"rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},{c[3]:.2f})"
                     if isinstance(c, tuple) else c for c in colors]

    # --- Build figure ---
    fig = go.Figure()

    # Bar layer
    if orientation == "horizontal":
        fig.add_trace(
            go.Bar(
                y=df[x],
                x=df[y],
                orientation="h",
                marker_color=colors,
                hovertemplate="<b>%{y}</b><br>Value: %{x}<extra></extra>",
            )
        )
    else:
        fig.add_trace(
            go.Bar(
                x=df[x],
                y=df[y],
                marker_color=colors,
                hovertemplate="<b>%{x}</b><br>Value: %{y}<extra></extra>",
            )
        )

    # Trace line (optional)
    if trace_line:
        if orientation == "horizontal":
            fig.add_trace(
                go.Scatter(
                    x=df[y],
                    y=df[x],
                    mode="lines+markers",
                    line=dict(color="grey", width=1.2),
                    marker=dict(color="grey", size=6),
                    name="trend",
                    hoverinfo="skip",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df[x],
                    y=df[y],
                    mode="lines+markers",
                    line=dict(color="grey", width=1.2),
                    marker=dict(color="grey", size=6),
                    name="trend",
                    hoverinfo="skip",
                )
            )

    # Value annotations with white-bordered backgrounds
    if show_values:
        if orientation == "horizontal":
            for i, val in enumerate(df[y]):
                fig.add_annotation(
                    x=val,
                    y=df[x].iloc[i],
                    text=f"<b>{val:.2f}</b>",
                    showarrow=False,
                    xanchor="left",
                    yanchor="middle",
                    font=dict(size=10, color="black", family="Arial Black"),
                    bgcolor="rgba(255, 255, 255, 0.95)",
                    bordercolor="#333333",
                    borderwidth=1.5,
                    borderpad=4,
                    xshift=10,
                )
        else:
            for i, val in enumerate(df[y]):
                fig.add_annotation(
                    x=df[x].iloc[i],
                    y=val,
                    text=f"<b>{val:.2f}</b>",
                    showarrow=False,
                    yanchor="bottom",
                    font=dict(size=10, color="black", family="Arial Black"),
                    bgcolor="rgba(255, 255, 255, 0.95)",
                    bordercolor="#333333",
                    borderwidth=1.5,
                    borderpad=4,
                    yshift=8,
                )

    # Layout styling
    title_dict = {}
    if subtitle:
        # Top-left corner when subtitle present
        title_dict = dict(
            text=f"<b>{title or f'{y.title()} by {x.title()}'}</b>"
                 + f"<br><span style='color:grey;font-size:14px;'>{subtitle}</span>",
            x=0.02,
            xanchor="left",
            yanchor="top",
            y=0.96,
        )
    else:
        # Centered when no subtitle
        title_dict = dict(
            text=f"<b>{title or f'{y.title()} by {x.title()}'}</b>",
            x=0.5,
            xanchor="center",
            yanchor="top",
            y=0.96,
        )

    fig.update_layout(
        template="plotly_white",
        height=600,
        margin=dict(t=90, b=50, l=60, r=30),
        title=title_dict,
        xaxis_title=dict(
            text=x.title() if orientation != "horizontal" else y.title(),
            font=dict(size=12, color="black", family="Arial, sans-serif"),
        ),
        yaxis_title=dict(
            text=y.title() if orientation != "horizontal" else "",
            font=dict(size=12, color="black", family="Arial, sans-serif"),
        ),
        plot_bgcolor="white",
        showlegend=False,
    )

    # Fine-tuning axis fonts
    fig.update_xaxes(showgrid=False, tickfont=dict(size=11, color="#333"))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(180,180,180,0.3)", tickfont=dict(size=11, color="#333"))

    return fig
