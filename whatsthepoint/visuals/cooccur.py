"""
cooccur.py: whatsthepoint Co-occurrence Network Visualization
---------------------------------------------------------------
Text co-occurrences drawn as static or interactive networks.

Features:
- Centrality-based node shading (viridis)
- Labels beside their nodes, never on top of them
- Compact title/subtitle handling
- Dynamic spacing for multi-subplot figures
"""

import ast
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp
from itertools import combinations
from collections import Counter
from matplotlib.colors import Normalize
from matplotlib import cm

from ..utils.style import set_style, apply_titles, format_tick_labels, is_house
from ..utils.layout import pad_limits, use_constrained
from ..utils.optional import require


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _parse_terms(x):
    """Safely parse comma- or list-like strings into Python lists."""
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return [t.strip() for t in x.split(",") if t.strip()]
    return []


def _compute_centrality(G):
    """Compute normalized betweenness centrality as shading metric."""
    nx = require("networkx", "network")
    return nx.betweenness_centrality(G, weight="weight", normalized=True)


# ══════════════════════════════════════════════════════════════════════════════
# STATIC VERSION
# ══════════════════════════════════════════════════════════════════════════════
def cooccur(
    df,
    term_col,
    top_n=10,
    exclude=None,
    max_neighbors=10,
    style="house",
    figsize=(10, 7),
    density_mode=False,
):
    """
    whatsthepoint static co-occurrence network plot, one panel per central term.

    Returns the matplotlib Figure (None when no terms co-occur).
    """
    set_style(style)
    df = df.copy()
    df["term_list"] = df[term_col].apply(_parse_terms)
    if exclude:
        excl = {t.lower() for t in exclude}
        df["term_list"] = df["term_list"].apply(
            lambda lst: [t for t in lst if t.lower() not in excl]
        )

    all_terms = [t for lst in df["term_list"] for t in lst]
    term_counts = Counter(all_terms)
    edge_weights = Counter()
    for terms in df["term_list"]:
        for u, v in combinations(sorted(set(terms)), 2):
            edge_weights[(u, v)] += 1
    if not edge_weights:
        print("No valid co-occurrences found.")
        return None

    # Build global co-occurrence network to compute centralities
    nx = require("networkx", "network")
    G_global = nx.Graph()
    for (u, v), w in edge_weights.items():
        G_global.add_edge(u, v, weight=w)
    centrality = _compute_centrality(G_global)
    top_terms = sorted(centrality, key=centrality.get, reverse=True)[:top_n]

    # Setup figure
    house = is_house(style)
    fig, axes = plt.subplots(
        nrows=top_n, ncols=1,
        figsize=(figsize[0], figsize[1] * top_n),
        constrained_layout=False,
    )
    if top_n == 1:
        axes = [axes]
    fig.set_facecolor("white")
    if house:
        use_constrained(fig)
    else:
        # Compact global title block - position top left
        fig.text(
            0.02, 0.995,
            f"Co-occurrence Networks for Top {top_n} Central Terms",
            fontsize=16, fontweight="bold", color="#111111", ha="left", va="top",
        )
        fig.text(
            0.02, 0.982,
            f"Nodes shaded by betweenness centrality (showing top {max_neighbors} neighbors)",
            ha="left", va="top", fontsize=11, color="grey",
        )

    cmap = matplotlib.colormaps["viridis"]
    for ax, central in zip(axes, top_terms):
        ax.axis("off")
        neighbors = [
            (v if u == central else u, w)
            for (u, v), w in edge_weights.items()
            if central in (u, v)
        ]
        if not neighbors:
            ax.text(0.5, 0.5, f"No links for '{central}'", ha="center", va="center")
            continue

        top_neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)[:max_neighbors]
        G = nx.Graph()
        G.add_node(central, size=term_counts[central], central=True)
        for n, w in top_neighbors:
            G.add_node(n, size=term_counts[n], central=False)
            G.add_edge(central, n, weight=w)

        node_centralities = _compute_centrality(G)
        if house:
            # the central term in the middle, its neighbours evenly round it,
            # strongest link at the top and then clockwise: no two nodes can meet
            pos = {central: np.array([0.0, 0.0])}
            for k, (n, _) in enumerate(top_neighbors):
                angle = np.pi / 2 - 2 * np.pi * k / len(top_neighbors)
                pos[n] = np.array([np.cos(angle), np.sin(angle)])
        else:
            pos = nx.spring_layout(G, k=0.7 if density_mode else 1.0, iterations=40, seed=42)
        # Increase node sizes dynamically based on text length
        node_sizes = []
        for node, data in G.nodes(data=True):
            text_len = len(str(node))
            # Base size on logarithm of count plus text length factor
            base_size = np.log(data["size"] + 2) * 600
            # Add extra size based on text length (minimum 50 per character)
            text_size = text_len * 50
            node_sizes.append(max(base_size, text_size))

        if house:
            node_colors = ["#CCCCCC" if n == central else "#FFFFFF" for n in G.nodes()]
        else:
            node_colors = [cmap(node_centralities.get(n, 0.1)) for n in G.nodes()]
        weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
        edge_widths = 0.5 + 5 * (weights / weights.max())

        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                               edge_color="#000000" if house else "grey", alpha=1.0 if house else 0.35)
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=node_sizes, node_color=node_colors,
            alpha=1.0 if house else 0.9, edgecolors="#000000" if house else "white",
            linewidths=1.2 if house else 1.0,
        )

        # Labels sit beside their node, on the side away from the central term,
        # so no label covers a node. The central term is labelled below itself
        # when no neighbour is there, otherwise in the widest gap between links.
        centre = np.asarray(pos[central], dtype=float)
        others = [np.asarray(pos[n], dtype=float) - centre for n in G.nodes() if n != central]
        angles = sorted(np.arctan2(d[1], d[0]) for d in others)
        gaps = [(b - a, (a + b) / 2) for a, b in zip(angles, angles[1:] + [angles[0] + 2 * np.pi])]
        widest = max(gaps)[1] if gaps else -np.pi / 2
        for (node, xy), size in zip(pos.items(), node_sizes):
            radius = np.sqrt(size) / 2 + 5          # points, from the node centre
            if node == central:
                direction = np.array([np.cos(widest), np.sin(widest)])
            else:
                direction = np.asarray(xy, dtype=float) - centre
                direction = direction / (np.hypot(*direction) or 1.0)
            ax.annotate(
                node, xy=xy, xytext=tuple(radius * direction), textcoords="offset points",
                ha="left" if direction[0] > 0.35 else "right" if direction[0] < -0.35 else "center",
                va="bottom" if direction[1] > 0.35 else "top" if direction[1] < -0.35 else "center",
                fontsize=10, color="#111111", fontweight="bold", zorder=5, annotation_clip=False,
                bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.15"),
            )

        ax.margins(0.12)
        if house:
            # a panel title only when there are several panels to tell apart
            if top_n > 1:
                ax.set_title(f"{central}: top {len(top_neighbors)} co-occurring terms",
                             fontsize=12, fontweight="bold", color="#000000", pad=6)
            pad_limits(ax, margin=6)
        else:
            ax.set_title(f"'{central}': Top {max_neighbors} Neighbors",
                         fontsize=12, fontweight="bold", color="#333333", pad=6)

    if not house:
        plt.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE VERSION
# ══════════════════════════════════════════════════════════════════════════════
def cooccur_interactive(
    df,
    term_col,
    top_n=5,
    exclude=None,
    max_neighbors=8,
    style="viridis",
    width=900,
    height_per_network=500,
):
    """whatsthepoint interactive co-occurrence visualization (centrality-colored)."""
    set_style(style)
    df = df.copy()
    df["term_list"] = df[term_col].apply(_parse_terms)
    if exclude:
        excl = {t.lower() for t in exclude}
        df["term_list"] = df["term_list"].apply(lambda lst: [t for t in lst if t.lower() not in excl])

    all_terms = [t for lst in df["term_list"] for t in lst]
    term_counts = Counter(all_terms)
    edge_weights = Counter()
    for terms in df["term_list"]:
        for u, v in combinations(sorted(set(terms)), 2):
            edge_weights[(u, v)] += 1
    if not edge_weights:
        raise ValueError("No co-occurrence edges found.")

    # Global centrality
    nx = require("networkx", "network")
    G_global = nx.Graph()
    for (u, v), w in edge_weights.items():
        G_global.add_edge(u, v, weight=w)
    centrality = _compute_centrality(G_global)
    top_terms = sorted(centrality, key=centrality.get, reverse=True)[:top_n]

    fig = sp.make_subplots(rows=top_n, cols=1, vertical_spacing=0.08)
    cmap = matplotlib.colormaps["viridis"]

    for i, central in enumerate(top_terms, start=1):
        neighbors = [(v if u == central else u, w)
                     for (u, v), w in edge_weights.items()
                     if central in (u, v)]
        top_neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)[:max_neighbors]
        if not top_neighbors:
            continue

        G = nx.Graph()
        G.add_node(central, size=term_counts[central], group="central")
        for n, w in top_neighbors:
            G.add_node(n, size=term_counts[n], group="neighbor")
            G.add_edge(central, n, weight=w)
        node_centralities = _compute_centrality(G)
        pos = nx.spring_layout(G, k=0.9, seed=42)

        # Edges
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]; x1, y1 = pos[v]
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
        fig.add_trace(
            go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(width=0.8, color="grey"), opacity=0.4, hoverinfo="none",
            ),
            row=i, col=1
        )

        # Nodes - dynamic sizing based on text length
        xs, ys, texts, sizes, colors = [], [], [], [], []
        for node, data in G.nodes(data=True):
            x, y = pos[node]
            xs.append(x); ys.append(y)
            texts.append(f"<b>{node}</b><br>Centrality: {node_centralities.get(node, 0):.2f}")
            # Increase node sizes dynamically based on text length
            text_len = len(str(node))
            base_size = 18 + np.log(data["size"] + 1) * 25
            text_size = text_len * 3  # 3 pixels per character
            sizes.append(max(base_size, text_size))
            c = cmap(node_centralities.get(node, 0))
            colors.append(f"rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, {c[3]:.2f})")

        # Add nodes with white background labels
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys,
                mode="markers",
                hovertext=texts,
                marker=dict(size=sizes, color=colors, line=dict(color="white", width=1)),
                hoverinfo="text",
                showlegend=False,
            ),
            row=i, col=1
        )

        # Add text annotations with white backgrounds
        for node, (x_pos, y_pos) in zip(G.nodes(), zip(xs, ys)):
            fig.add_annotation(
                x=x_pos, y=y_pos,
                text=f"<b>{node}</b>",
                showarrow=False,
                font=dict(size=11, color="#111111"),
                bgcolor="white",
                bordercolor="grey",
                borderwidth=1.5,
                borderpad=4,
                opacity=0.9,
                row=i, col=1
            )

        fig.update_yaxes(visible=False, row=i, col=1)
        fig.update_xaxes(visible=False, row=i, col=1)

    fig.update_layout(
        height=height_per_network * top_n,
        width=width,
        title=dict(
            text=f"<b>Co-occurrence Networks (Top {top_n} Central Terms)</b>"
                 f"<br><span style='color:grey;font-size:14px;'>Nodes shaded by betweenness centrality, top {max_neighbors} neighbors</span>",
            x=0.5, xanchor="center", yanchor="top", y=0.98,
        ),
        margin=dict(l=50, r=50, t=80, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
    )
    return fig
