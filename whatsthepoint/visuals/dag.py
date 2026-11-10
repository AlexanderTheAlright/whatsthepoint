"""
dag.py: whatsthepoint Causal DAG Visualizations
------------------------------------------------
Directed Acyclic Graphs for causal inference.

Features:
- DAG visualization with networkx
- Confounding path highlighting
- Multiple layout algorithms
- Interactive Plotly version

Note: Requires networkx
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional, List, Dict, Tuple


try:
    import networkx as nx
except ImportError:
    nx = None

try:
    from ..utils.style import (
        set_style,
        generate_semantic_palette,
        apply_titles,
        COLORS_DICT,
        get_color,
        format_tick_labels,
        is_house,
    )
    from ..utils.layout import (
        EDGE_LABEL_GID, pad_limits, use_constrained, wrap_label,
        _renderer, _text_box, _inside, _overlap,
    )
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.transforms import Bbox
except ImportError:
    def set_style(*args, **kwargs):
        pass
    def apply_titles(*args, **kwargs):
        pass
    def get_color(*args, **kwargs):
        return '#333333'
    COLORS_DICT = {'viridis': plt.cm.viridis}


# ==============================================================================
# DAG
# ==============================================================================

def dag(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    layout: str = "hierarchical",
    highlight_nodes: Optional[List[str]] = None,
    highlight_edges: Optional[List[Tuple[str, str]]] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "house",
    figsize: tuple = (10, 8),
    output_path: Optional[str] = None,
):
    """
    Create a directed acyclic graph (DAG) for causal inference.

    Parameters
    ----------
    nodes : list of str
        List of node names (variables).
    edges : list of tuples
        List of directed edges as (source, target) tuples. A third element,
        (source, target, label), prints the label on the edge.
    layout : str, default "hierarchical"
        Layout algorithm: 'hierarchical', 'circular', 'spring', 'shell'.
    highlight_nodes : list of str, optional
        Nodes to highlight (e.g., treatment and outcome).
    highlight_edges : list of tuples, optional
        Edges to highlight (e.g., confounding paths).
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    style_mode : str, default "house"
        Color scheme.
    figsize : tuple, default (10, 8)
        Figure size.
    output_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> nodes = ['Treatment', 'Outcome', 'Confounder', 'Mediator']
    >>> edges = [('Treatment', 'Outcome'), ('Treatment', 'Mediator'),
    ...          ('Mediator', 'Outcome'), ('Confounder', 'Treatment'),
    ...          ('Confounder', 'Outcome')]
    >>> dag(nodes, edges, highlight_nodes=['Treatment', 'Outcome'],
    ...     title='Causal DAG')
    """
    if nx is None:
        raise ImportError("networkx is required for DAG visualizations. Install it with: pip install whatsthepoint[network]")

    set_style(style_mode)

    edge_labels = {(e[0], e[1]): str(e[2]) for e in edges if len(e) > 2}
    edges = [(e[0], e[1]) for e in edges]

    # Create graph
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Choose layout with improved hierarchical positioning
    if layout == "hierarchical":
        # Try graphviz dot layout first (best for DAGs)
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=LR')
            # Normalize positions for better display
            if pos:
                x_coords = [p[0] for p in pos.values()]
                y_coords = [p[1] for p in pos.values()]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                # Normalize to reasonable range
                for node in pos:
                    x, y = pos[node]
                    pos[node] = (
                        (x - x_min) / (x_max - x_min + 0.001) * 3,
                        (y - y_min) / (y_max - y_min + 0.001) * 2
                    )
        except:
            # Fallback to custom hierarchical layout
            try:
                # Column-based layout with topological ordering
                # Identify node types by name
                treatment_nodes = [n for n in nodes if 'treatment' in n.lower()]
                outcome_nodes = [n for n in nodes if 'outcome' in n.lower()]
                mediator_nodes = [n for n in nodes if 'mediator' in n.lower() or 'mediat' in n.lower()]
                moderator_nodes = [n for n in nodes if 'moderator' in n.lower() or 'moderat' in n.lower()]
                confounder_nodes = [n for n in nodes if 'confounder' in n.lower() or 'confound' in n.lower()]
                collider_nodes = [n for n in nodes if 'collider' in n.lower() or 'collid' in n.lower()]

                # Assign nodes to layers based on topological order
                # Layer 0: Treatment/Source nodes (no predecessors or explicitly named)
                # Layer 1: Confounders (affect both treatment and outcome)
                # Layer 2: Mediators/Moderators (between treatment and outcome)
                # Layer 3: Colliders (affected by multiple nodes)
                # Layer 4: Outcomes/Sink nodes (no successors or explicitly named)

                layers = {0: [], 1: [], 2: [], 3: [], 4: []}

                # First, place explicitly named nodes
                for node in treatment_nodes:
                    layers[0].append(node)
                for node in outcome_nodes:
                    layers[4].append(node)
                for node in confounder_nodes:
                    layers[1].append(node)
                for node in mediator_nodes + moderator_nodes:
                    layers[2].append(node)
                for node in collider_nodes:
                    layers[3].append(node)

                # For unnamed nodes, use graph structure
                assigned = set(treatment_nodes + outcome_nodes + confounder_nodes +
                             mediator_nodes + moderator_nodes + collider_nodes)

                for node in nodes:
                    if node in assigned:
                        continue

                    in_deg = G.in_degree(node)
                    out_deg = G.out_degree(node)

                    # Source nodes (no predecessors)
                    if in_deg == 0:
                        layers[0].append(node)
                    # Sink nodes (no successors)
                    elif out_deg == 0:
                        layers[4].append(node)
                    # Intermediate nodes - use topological depth
                    else:
                        # Calculate shortest path from sources
                        sources = [n for n in G.nodes() if G.in_degree(n) == 0]
                        if sources:
                            try:
                                depths = []
                                for source in sources:
                                    if nx.has_path(G, source, node):
                                        depth = nx.shortest_path_length(G, source, node)
                                        depths.append(depth)

                                if depths:
                                    avg_depth = sum(depths) / len(depths)
                                    # Map depth to layer (1-3)
                                    if avg_depth <= 1:
                                        layers[1].append(node)
                                    elif avg_depth <= 2:
                                        layers[2].append(node)
                                    else:
                                        layers[3].append(node)
                                else:
                                    layers[2].append(node)
                            except:
                                layers[2].append(node)
                        else:
                            layers[2].append(node)

                # Create positions with proper spacing
                pos = {}
                x_spacing = 1.0  # Horizontal spacing between layers
                y_spacing = 0.8  # Vertical spacing between nodes in same layer

                for layer_idx, layer_nodes in layers.items():
                    if not layer_nodes:
                        continue

                    x = layer_idx * x_spacing

                    # Center nodes vertically
                    for i, node in enumerate(layer_nodes):
                        y = (len(layer_nodes) - 1) / 2.0 - i
                        pos[node] = (x, y * y_spacing)


            except Exception as e:
                # Final fallback to spring layout
                import traceback
                traceback.print_exc()
                print(f"Warning: Custom hierarchical layout failed ({e}), using spring layout")
                pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "spring":
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    else:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis('off')
    house = is_house(style_mode)

    # Get colors
    if house:
        edge_color = node_edge = highlight_edge = warning = "#000000"
        highlight_fill = "#D9D9D9"
    else:
        color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
        node_edge, edge_color, highlight_edge = color_map(0.6), color_map(0.3), color_map(0.9)
        warning = get_color('warning', style_mode)
        highlight_fill = "white"

    if title or subtitle:
        apply_titles(fig, title=title, subtitle=subtitle)
    use_constrained(fig)

    # Each node is drawn as its own label box, so a node can never sit hidden
    # under its label, and its size always fits its text.
    boxes = {}
    for node in G.nodes():
        x, y = pos[node]
        marked = bool(highlight_nodes and node in highlight_nodes)
        boxes[node] = ax.text(
            x, y, wrap_label(node), fontsize=11, fontweight='bold', color='#111111',
            ha='center', va='center', zorder=10,
            bbox=dict(boxstyle='round,pad=0.6',
                      facecolor=highlight_fill if marked else 'white',
                      edgecolor=highlight_edge if marked else node_edge,
                      linewidth=2.4 if marked else 1.2))

    if boxes:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        span_x = (max(xs) - min(xs)) or 1.0
        span_y = (max(ys) - min(ys)) or 1.0
        ax.set_xlim(min(xs) - 0.08 * span_x, max(xs) + 0.08 * span_x)
        ax.set_ylim(min(ys) - 0.25 * span_y, max(ys) + 0.25 * span_y)
        pad_limits(ax, margin=8)
        _draw_edges(ax, G, pos, boxes, edge_labels, highlight_edges, edge_color, warning)

    if output_path:
        from ..utils.style import save_figure
        save_figure(fig, output_path)

    return fig


def _arc(a, b, rad, n=60):
    """Points along the curve matplotlib draws for 'arc3,rad=rad' from a to b (pixels)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mid = (a + b) / 2.0
    d = b - a
    c = mid + rad * np.array([d[1], -d[0]])
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * c + t ** 2 * b


def _draw_edges(ax, G, pos, boxes, edge_labels, highlight_edges, edge_color, warning):
    """
    Draw every edge as an arrow from box edge to box edge. An edge that would
    run behind another node is bent until it clears it. Edge labels sit on
    their own edge, clear of every node, every other label and every other edge.
    """
    fig = ax.figure
    gap = 8 * fig.dpi / 72.0

    for attempt in range(6):
        for old in [p for p in ax.patches if isinstance(p, FancyArrowPatch)]:
            old.remove()
        for old in [t for t in ax.texts if t.get_gid() == EDGE_LABEL_GID]:
            old.remove()
        renderer = _renderer(fig)
        frame = ax.get_window_extent(renderer)
        extent = {n: _text_box(t, renderer) for n, t in boxes.items()}
        centre = {n: ax.transData.transform(pos[n]) for n in boxes}

        curves = {}
        for u, v in G.edges():
            others = [extent[n] for n in boxes if n not in (u, v)]
            best = None
            for rad in (0.0, 0.18, -0.18, 0.32, -0.32, 0.46, -0.46, 0.6, -0.6):
                pts = _arc(centre[u], centre[v], rad)
                hits = sum(_inside(bb, x, y, grow=gap) for bb in others for x, y in pts)
                if best is None or hits < best[0]:
                    best = (hits, rad, pts)
                if hits == 0:
                    break
            curves[(u, v)] = best[1:]

        for (u, v), (rad, pts) in curves.items():
            marked = bool(highlight_edges and (u, v) in highlight_edges)
            ax.add_patch(FancyArrowPatch(
                posA=pos[u], posB=pos[v],
                patchA=boxes[u].get_bbox_patch(), patchB=boxes[v].get_bbox_patch(),
                connectionstyle=f"arc3,rad={rad}", arrowstyle='-|>', mutation_scale=16,
                shrinkA=3, shrinkB=3, linewidth=2.6 if marked else 1.3,
                color=warning if marked else edge_color,
                zorder=11,   # above the boxes, so each box is placed before its arrows are cut to it
            ))

        placed = []
        for (u, v), (rad, pts) in curves.items():
            if (u, v) not in edge_labels:
                continue
            # the part of the curve that shows between the two boxes
            shown = [(x, y) for x, y in pts
                     if not _inside(extent[u], x, y, grow=4) and not _inside(extent[v], x, y, grow=4)]
            if not shown:
                continue
            probe = ax.text(0, 0, edge_labels[(u, v)], fontsize=10, fontweight='bold')
            size = probe.get_window_extent(renderer)
            probe.remove()
            half_w, half_h = size.width / 2 + 5, size.height / 2 + 5
            rivals = [q for e, (_, other) in curves.items() if e != (u, v) for q in other]
            spot = shown[len(shown) // 2]
            for share in (0.5, 0.4, 0.6, 0.3, 0.7, 0.22, 0.78):
                x, y = shown[min(int(share * len(shown)), len(shown) - 1)]
                box = Bbox.from_extents(x - half_w, y - half_h, x + half_w, y + half_h)
                free = (not any(_overlap(box, bb, 0) for bb in extent.values())
                        and not any(_overlap(box, bb, 0) for bb in placed)
                        and not any(_inside(box, qx, qy, grow=gap) for qx, qy in rivals))
                if free:
                    spot = (x, y)
                    placed.append(box)
                    break
            dx, dy = ax.transData.inverted().transform(spot)
            ax.text(dx, dy, edge_labels[(u, v)], fontsize=10, fontweight='bold', color='#111111',
                    ha='center', va='center', zorder=12, gid=EDGE_LABEL_GID,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none'))

        # a bent edge may leave the axes: widen the limits and lay the edges out again
        every = [q for _, pts in curves.values() for q in pts]
        if all(_inside(frame, x, y, grow=-gap / 2) for x, y in every):
            break
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_xlim(x0 - 0.06 * (x1 - x0), x1 + 0.06 * (x1 - x0))
        ax.set_ylim(y0 - 0.10 * (y1 - y0), y1 + 0.10 * (y1 - y0))
    _renderer(fig)


def dag_from_formula(
    formula: str,
    layout: str = "hierarchical",
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "house",
    figsize: tuple = (10, 8),
    output_path: Optional[str] = None,
):
    """
    Create a DAG from a formula string.

    Parameters
    ----------
    formula : str
        Formula describing edges, e.g., "A -> B; A -> C; C -> B".
    layout : str, default "hierarchical"
        Layout algorithm.
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    style_mode : str, default "house"
        Color scheme.
    figsize : tuple, default (10, 8)
        Figure size.
    output_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> dag_from_formula("Treatment -> Outcome; Confounder -> Treatment; Confounder -> Outcome",
    ...                  title='DAG from Formula')
    """
    # Parse formula
    edges = []
    for edge_str in formula.split(';'):
        edge_str = edge_str.strip()
        if '->' in edge_str:
            source, target = edge_str.split('->')
            edges.append((source.strip(), target.strip()))

    # Extract nodes
    nodes = list(set([e[0] for e in edges] + [e[1] for e in edges]))

    return dag(
        nodes=nodes,
        edges=edges,
        layout=layout,
        title=title,
        subtitle=subtitle,
        style_mode=style_mode,
        figsize=figsize,
        output_path=output_path
    )


# ==============================================================================
# INTERACTIVE VERSION
# ==============================================================================

def dag_interactive(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    layout: str = "spring",
    highlight_nodes: Optional[List[str]] = None,
    highlight_edges: Optional[List[Tuple[str, str]]] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "house",
):
    """
    Interactive DAG using Plotly.

    Parameters
    ----------
    nodes : list of str
        List of node names.
    edges : list of tuples
        List of directed edges.
    layout : str, default "spring"
        Layout algorithm.
    highlight_nodes : list of str, optional
        Nodes to highlight.
    highlight_edges : list of tuples, optional
        Edges to highlight.
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    style_mode : str, default "house"
        Color scheme.

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    >>> dag_interactive(nodes, edges, highlight_nodes=['Treatment', 'Outcome'],
    ...                 title='Interactive Causal DAG')
    """
    if nx is None:
        raise ImportError("networkx is required. Install with: pip install networkx")

    set_style(style_mode)

    # Create graph
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Get layout with improved hierarchical positioning (same as static version)
    if layout == "hierarchical":
        # Try graphviz dot layout first (best for DAGs)
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=LR')
            # Normalize positions for better display
            if pos:
                x_coords = [p[0] for p in pos.values()]
                y_coords = [p[1] for p in pos.values()]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                # Normalize to reasonable range
                for node in pos:
                    x, y = pos[node]
                    pos[node] = (
                        (x - x_min) / (x_max - x_min + 0.001) * 3,
                        (y - y_min) / (y_max - y_min + 0.001) * 2
                    )
        except:
            # Fallback to custom hierarchical layout
            try:
                # Column-based layout with topological ordering
                treatment_nodes = [n for n in nodes if 'treatment' in n.lower()]
                outcome_nodes = [n for n in nodes if 'outcome' in n.lower()]
                mediator_nodes = [n for n in nodes if 'mediator' in n.lower() or 'mediat' in n.lower()]
                moderator_nodes = [n for n in nodes if 'moderator' in n.lower() or 'moderat' in n.lower()]
                confounder_nodes = [n for n in nodes if 'confounder' in n.lower() or 'confound' in n.lower()]
                collider_nodes = [n for n in nodes if 'collider' in n.lower() or 'collid' in n.lower()]

                # Assign nodes to layers
                layers = {0: [], 1: [], 2: [], 3: [], 4: []}

                # Place explicitly named nodes
                for node in treatment_nodes:
                    layers[0].append(node)
                for node in outcome_nodes:
                    layers[4].append(node)
                for node in confounder_nodes:
                    layers[1].append(node)
                for node in mediator_nodes + moderator_nodes:
                    layers[2].append(node)
                for node in collider_nodes:
                    layers[3].append(node)

                # For unnamed nodes, use graph structure
                assigned = set(treatment_nodes + outcome_nodes + confounder_nodes +
                             mediator_nodes + moderator_nodes + collider_nodes)

                for node in nodes:
                    if node in assigned:
                        continue

                    in_deg = G.in_degree(node)
                    out_deg = G.out_degree(node)

                    if in_deg == 0:
                        layers[0].append(node)
                    elif out_deg == 0:
                        layers[4].append(node)
                    else:
                        sources = [n for n in G.nodes() if G.in_degree(n) == 0]
                        if sources:
                            try:
                                depths = []
                                for source in sources:
                                    if nx.has_path(G, source, node):
                                        depth = nx.shortest_path_length(G, source, node)
                                        depths.append(depth)

                                if depths:
                                    avg_depth = sum(depths) / len(depths)
                                    if avg_depth <= 1:
                                        layers[1].append(node)
                                    elif avg_depth <= 2:
                                        layers[2].append(node)
                                    else:
                                        layers[3].append(node)
                                else:
                                    layers[2].append(node)
                            except:
                                layers[2].append(node)
                        else:
                            layers[2].append(node)

                # Create positions
                pos = {}
                x_spacing = 1.0
                y_spacing = 0.8

                for layer_idx, layer_nodes in layers.items():
                    if not layer_nodes:
                        continue
                    x = layer_idx * x_spacing
                    for i, node in enumerate(layer_nodes):
                        y = (len(layer_nodes) - 1) / 2.0 - i
                        pos[node] = (x, y * y_spacing)


            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Warning: Custom hierarchical layout failed ({e}), using spring layout")
                pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "spring":
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    else:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Create edge traces
    edge_traces = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        is_highlighted = highlight_edges and edge in highlight_edges

        highlight_edge_color = get_color('warning', style_mode)
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(
                width=3 if is_highlighted else 2,
                color=highlight_edge_color if is_highlighted else 'gray'
            ),
            hoverinfo='none',
            showlegend=False,
        )
        edge_traces.append(edge_trace)

    # Create node trace
    node_x = []
    node_y = []
    node_colors = []
    node_text = []

    highlight_node_color = get_color('highlight', style_mode)
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

        is_highlighted = highlight_nodes and node in highlight_nodes
        node_colors.append(highlight_node_color if is_highlighted else 'steelblue')

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        marker=dict(
            size=50,
            color=node_colors,
            line=dict(width=2, color='black')
        ),
        hoverinfo='text',
        hovertext=node_text,
        showlegend=False,
    )

    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])

    # Add text annotations with white backgrounds for better readability
    for i, node in enumerate(G.nodes()):
        x, y = pos[node]
        fig.add_annotation(
            x=x,
            y=y,
            text=node,
            showarrow=False,
            font=dict(size=13, color='black', family='Arial Black'),
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='#333333',
            borderwidth=1.5,
            borderpad=4,
        )

    # Layout
    title_text = f"<b>{title or 'Causal DAG'}</b>"
    if subtitle:
        title_text += f"<br><span style='color:grey;font-size:14px;'>{subtitle}</span>"

    fig.update_layout(
        template="plotly_white",
        height=700,
        margin=dict(t=90, b=30, l=30, r=30),
        title=dict(text=title_text, x=0.5, xanchor="center"),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
    )

    return fig
