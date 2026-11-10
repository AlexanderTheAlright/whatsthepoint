"""
scatter.py: whatsthepoint Visualization Module
----------------------------------------------
Semantic scatterplots with optional grouping and regression lines.
Supports dynamic theme switching from whatsthepoint style engine.

Features:
- Static and interactive versions
- Group-based coloring
- Regression lines with a 95% prediction band (fitted value +/- 1.96 SD of residuals)
- LOWESS smoothing
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import linregress
from matplotlib import cm

# Correct relative import (visuals → utils)
from ..utils.style import (
    set_style,
    apply_titles,
    generate_semantic_palette,
    format_tick_labels,
    legend_below,
    is_house,
    BW_LINESTYLES,
    BW_MARKERS,
    BW_MARKER_FILLS,
)
from ..utils.layout import finish, sentence_label
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import patheffects


# ══════════════════════════════════════════════════════════════════════════════
# SCATTER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def scatterplot(
    df,
    x,
    y,
    group=None,
    title=None,
    subtitle=None,
    palette=None,
    ci=True,
    line=True,
    smooth=False,
    legend_title=None,
    alpha=0.8,
    s=50,
    figsize=(10, 6),
    style_mode="house",
):
    """
    whatsthepoint scatter plot with semantic colour logic and regression overlays.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    x, y : str
        Variables to plot.
    group : str, optional
        Column used to colour points by group.
    title, subtitle : str, optional
        Plot titles.
    palette : dict, optional
        Custom colour mapping; auto-generated if None.
    ci : bool, default True
        Shade a 95% prediction band around each linear fit: the fitted value
        plus or minus 1.96 times the standard deviation of the residuals. It
        shows where about 95% of observations fall, not the uncertainty of
        the fitted line, so it is not a confidence interval. Ignored when
        ``smooth=True``.
    line : bool, default True
        Draw regression line(s).
    smooth : bool, default False
        Apply LOWESS smoothing.
    legend_title : str, optional
        Legend title. None by default.
    alpha : float, default 0.8
        Marker transparency.
    s : int, default 50
        Marker size.
    figsize : tuple, default (10, 6)
        Figure size.
    style_mode : str, default "house"
        One of {'house','viridis','reviewer3','sentiment','fiery','plainjane'}.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    # Activate style
    set_style(style_mode)
    fig, ax = plt.subplots(figsize=figsize, dpi=130)

    # ─── Palette logic ─────────────────────────────────────────────────────────
    if group and palette is None:
        # Filter out groups with no meaningful data
        def has_meaningful_data(group_df, col_x, col_y):
            vals_x = group_df[col_x].dropna()
            vals_y = group_df[col_y].dropna()
            # Need at least 2 points for meaningful scatter
            if len(vals_x) < 2 or len(vals_y) < 2:
                return False
            # Check if all values are effectively zero
            if (vals_x.abs() < 1e-10).all() and (vals_y.abs() < 1e-10).all():
                return False
            return True

        all_groups = df[group].dropna().unique().tolist()
        groups = [g for g in all_groups if has_meaningful_data(df[df[group] == g], x, y)]
        # Put all groups in "positive" category to ensure maximum color contrast
        # This uses the discrete high-contrast palettes for 2-4 items
        g_dict = {"positive": groups}
        palette = generate_semantic_palette(g_dict, mode=style_mode)
    elif palette is None:
        # Default single tone by style
        import matplotlib.cm as cm
        palette = {"default": (
            cm.plasma(0.7) if style_mode == "fiery" else
            matplotlib.colormaps["viridis"](0.7) if style_mode == "viridis" else
            cm.Greens(0.7) if style_mode == "sentiment" else
            cm.Blues(0.7) if style_mode == "plainjane" else
            cm.Greys(0.5)
        )}

    # ─── Plot points and lines ────────────────────────────────────────────────
    band_drawn = False
    # Black outlines in reviewer3 keep white markers visible on a white page
    point_edge = "black" if style_mode in ("reviewer3", "house") else "white"
    house = is_house(style_mode)
    house_handles = []
    # a white casing keeps a black fitted line readable where it crosses points
    casing = [patheffects.Stroke(linewidth=5.0, foreground="white"), patheffects.Normal()]
    if group:
        for k, (g, dfg) in enumerate(df.groupby(group)):
            color = palette.get(g, "grey")
            if house:
                # groups differ by marker shape and fill, lines by dash pattern
                shape = BW_MARKERS[k % len(BW_MARKERS)]
                face = BW_MARKER_FILLS[k % len(BW_MARKER_FILLS)]
                dash = BW_LINESTYLES[k % len(BW_LINESTYLES)]
                ax.scatter(dfg[x], dfg[y], s=s, marker=shape, facecolor=face, edgecolor="#000000",
                           linewidth=0.9, label=str(g), zorder=3)
                house_handles.append(Line2D(
                    [], [], color="#000000", lw=2.5 if line else 0, linestyle=dash if line else "none",
                    marker=shape, markersize=7, markerfacecolor=face, markeredgecolor="#000000",
                    markeredgewidth=1.0, label=str(g)))
                color = "#000000"
            else:
                dash = "-"
                ax.scatter(
                    dfg[x], dfg[y],
                    s=s, alpha=alpha,
                    color=color,
                    edgecolor=point_edge,
                    linewidth=0.6,
                    label=str(g),
                    zorder=3,
                )

            if line and len(dfg) >= 3:
                xs = np.linspace(dfg[x].min(), dfg[x].max(), 200)
                if smooth:
                    fitted = lowess(dfg[y], dfg[x], frac=0.4, return_sorted=True)
                    fit_line, = ax.plot(fitted[:, 0], fitted[:, 1], color=color, lw=2.5 if house else 2,
                                        linestyle=dash, zorder=4)
                    if house:
                        fit_line.set_path_effects(casing)
                else:
                    slope, intercept, *_ = linregress(dfg[x], dfg[y])
                    y_pred = intercept + slope * xs
                    fit_line, = ax.plot(xs, y_pred, color=color, lw=2.5 if house else 2,
                                        linestyle=dash, zorder=4)
                    if house:
                        fit_line.set_path_effects(casing)
                    if ci:
                        resid = dfg[y] - (intercept + slope * dfg[x])
                        pred_band = 1.96 * np.std(resid)
                        band_drawn = True
                        ax.fill_between(
                            xs,
                            y_pred - pred_band,
                            y_pred + pred_band,
                            color="#888888" if house else color,
                            alpha=0.15,
                            linewidth=0,
                            zorder=2,
                        )

    else:
        if house:
            palette = {"default": "#000000"}
        ax.scatter(df[x], df[y], s=s, alpha=1.0 if house else alpha,
                   color="#FFFFFF" if house else palette["default"],
                   edgecolor=point_edge, linewidth=0.9 if house else 0.5, zorder=3)
        if line and len(df) >= 3:
            slope, intercept, *_ = linregress(df[x], df[y])
            xs = np.linspace(df[x].min(), df[x].max(), 200)
            y_pred = intercept + slope * xs
            fit_line, = ax.plot(xs, y_pred, color=palette["default"], lw=2.5 if house else 2, zorder=4)
            if house:
                fit_line.set_path_effects(casing)
            if ci:
                resid = df[y] - (intercept + slope * df[x])
                pred_band = 1.96 * np.std(resid)
                band_drawn = True
                ax.fill_between(xs, y_pred - pred_band, y_pred + pred_band,
                                color="#888888" if house else palette["default"], alpha=0.15,
                                linewidth=0, zorder=2)

    # Scatter axes keep matplotlib autoscaling: a scatter has no zero baseline.

    # ─── Axis aesthetics ──────────────────────────────────────────────────────
    name = sentence_label if house else (lambda s: s.replace("_", " ").title())
    ax.set_xlabel(name(x), fontsize=12 if house else 14, weight="bold", color="black")
    ax.set_ylabel(name(y), fontsize=12 if house else 14, weight="bold", color="black")
    ax.tick_params(axis='both', which='major', labelsize=10 if house else 12)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    # ─── Titles & Legend ──────────────────────────────────────────────────────
    apply_titles(fig, title, subtitle)

    # Format tick labels: bold and angled
    format_tick_labels(ax)

    # One legend below the axes: groups, plus the prediction band if drawn
    handles, labels = ax.get_legend_handles_labels()
    if band_drawn:
        handles.append(Patch(facecolor="grey", alpha=0.25, edgecolor="none"))
        labels.append("95% prediction band (\u00b11.96 SD of residuals)")
    if house:
        if group:
            handles = house_handles + handles[len(house_handles):]
        if band_drawn:
            handles[-1] = Patch(facecolor="#888888", alpha=0.3, edgecolor="none")
        finish(fig, handles=handles, labels=labels, legend_title=legend_title,
               legend=bool(group or band_drawn))
        return fig, ax

    if group or band_drawn:
        legend_below(ax, handles=handles, labels=labels, title=legend_title)

    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    return fig, ax


def scatterplot_interactive(
    df,
    x,
    y,
    group=None,
    title=None,
    subtitle="",
    line=True,
    style_mode="house",
):
    """Interactive Plotly scatterplot with regression lines."""
    set_style(style_mode)

    # Palette logic
    if group:
        groups = df[group].dropna().unique().tolist()
        thirds = max(1, len(groups) // 3)
        g_dict = {
            "positive": groups[:thirds],
            "neutral": groups[thirds : 2 * thirds],
            "negative": groups[2 * thirds :],
        }
        palette = generate_semantic_palette(g_dict, mode=style_mode)
        # Convert to hex
        palette_hex = {}
        for k, v in palette.items():
            if isinstance(v, tuple):
                palette_hex[k] = f"rgba({int(v[0]*255)},{int(v[1]*255)},{int(v[2]*255)},{v[3] if len(v)>3 else 1})"
            else:
                palette_hex[k] = v
        palette = palette_hex
    else:
        cmap = matplotlib.colormaps["viridis"]
        rgba = cmap(0.7)
        palette = {"default": f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]})"}

    fig = go.Figure()

    if group:
        for g, dfg in df.groupby(group):
            color = palette.get(g, "grey")

            # Scatter points
            fig.add_trace(go.Scatter(
                x=dfg[x], y=dfg[y],
                mode="markers",
                name=str(g),
                marker=dict(size=8, color=color, opacity=0.7, line=dict(color="white", width=0.5)),
            ))

            # Regression line
            if line and len(dfg) >= 3:
                slope, intercept, *_ = linregress(dfg[x], dfg[y])
                xs = np.linspace(dfg[x].min(), dfg[x].max(), 100)
                y_pred = intercept + slope * xs
                fig.add_trace(go.Scatter(
                    x=xs, y=y_pred,
                    mode="lines",
                    name=f"{g} (trend)",
                    line=dict(color=color, width=2),
                    showlegend=False,
                ))
    else:
        fig.add_trace(go.Scatter(
            x=df[x], y=df[y],
            mode="markers",
            marker=dict(size=8, color=palette["default"], opacity=0.7,
                       line=dict(color="white", width=0.5)),
            showlegend=False,
        ))

        # Regression line
        if line and len(df) >= 3:
            slope, intercept, *_ = linregress(df[x], df[y])
            xs = np.linspace(df[x].min(), df[x].max(), 100)
            y_pred = intercept + slope * xs
            fig.add_trace(go.Scatter(
                x=xs, y=y_pred,
                mode="lines",
                line=dict(color=palette["default"], width=2),
                showlegend=False,
            ))

    fig.update_layout(
        title=f"<b>{title}</b><br><span style='color:grey'>{subtitle}</span>",
        xaxis_title=x.replace("_", " ").title(),
        yaxis_title=y.replace("_", " ").title(),
        template="plotly_white",
        height=600,
        width=900,
        hovermode="closest",
    )

    return fig
