"""
wordcloud.py: whatsthepoint Visuals
--------------------------------------
Static and interactive wordclouds with consistent whatsthepoint style.

Supports:
- Matplotlib / WordCloud static clouds
- Plotly interactive scatter-style clouds
- Semantic or frequency-based color schemes
- Highlight groups for tagged keywords
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from ..utils.style import set_style, apply_titles
from ..utils.optional import require
import plotly.graph_objects as go
from matplotlib import cm


# ══════════════════════════════════════════════════════════════════════════════
# WORDCLOUD
# ══════════════════════════════════════════════════════════════════════════════

def wordcloud(
    freq_dict,
    title=None,
    subtitle=None,
    style_mode="house",
    cmap=None,
    highlight_groups=None,
    use_gradient=False,
    use_custom_freq_colors=False,
    low_freq_color="#D3D3D3",
    high_freq_color="#000000",
    figsize=(10, 8),
    max_words=200,
    background="white",
    contour=False,
    contour_color="#000000",
):
    """
    Generate a whatsthepoint styled static wordcloud.

    Parameters
    ----------
    freq_dict : dict
        Mapping {word: frequency}.
    title, subtitle : str
        Plot title and subtitle.
    style_mode : str
        One of {'house', 'viridis', 'reviewer3', 'sentiment', 'fiery', 'plainjane'}.
    cmap : str
        Optional matplotlib colormap override.
    highlight_groups : dict
        {'GroupName': {'color': '#hex', 'words': [...]}}.
    use_gradient : bool
        Whether to shade colors by frequency.
    use_custom_freq_colors : bool
        Whether to blend between two custom colors by frequency.
    low_freq_color, high_freq_color : str
        Hex colors for low/high frequencies.
    figsize : tuple
        Figure size in inches.
    max_words : int
        Max number of words to show.
    """
    set_style(style_mode)
    if not freq_dict:
        raise ValueError("No frequencies provided.")

    cmap_obj = matplotlib.colormaps[cmap or "viridis"]
    max_f = max(freq_dict.values())
    min_f = min(freq_dict.values())

    def blend_hex(hex1, hex2, t):
        h1, h2 = hex1.lstrip("#"), hex2.lstrip("#")
        r1, g1, b1 = int(h1[0:2], 16), int(h1[2:4], 16), int(h1[4:6], 16)
        r2, g2, b2 = int(h2[0:2], 16), int(h2[2:4], 16), int(h2[4:6], 16)
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        return f"rgb({r},{g},{b})"

    def color_func(word, **kwargs):
        word_l = word.lower()
        if highlight_groups:
            for g, info in highlight_groups.items():
                if word_l in [w.lower() for w in info["words"]]:
                    return info["color"]

        freq = freq_dict.get(word, min_f)
        t = (freq - min_f) / (max_f - min_f + 1e-9)
        if use_custom_freq_colors:
            return blend_hex(low_freq_color, high_freq_color, t)
        elif use_gradient:
            r, g, b, _ = cmap_obj(t)
            return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
        else:
            return f"rgb({int(np.random.rand()*255)}, {int(np.random.rand()*255)}, {int(np.random.rand()*255)})"

    WordCloud = require("wordcloud", "text").WordCloud
    wc = WordCloud(
        width=1600,
        height=1200,
        background_color=background,
        color_func=color_func,
        max_words=max_words,
        collocations=False,
        contour_width=2 if contour else 0,
        contour_color=contour_color,
    ).generate_from_frequencies(freq_dict)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    apply_titles(fig, title, subtitle, n=len(freq_dict))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE WORDCLOUD (Plotly Scatter)
# ══════════════════════════════════════════════════════════════════════════════

def wordcloud_interactive(
    freq_dict,
    title=None,
    style_mode="house",
    highlight_groups=None,
    colormap="viridis",
    exact_words=200,
    spacing_factor=600,
    max_attempts=50,
    r_min=30,
    epsilon=15,
    use_gradient=True,
    use_custom_freq_colors=False,
    low_freq_color="#D3D3D3",
    high_freq_color="#000000",
    figsize=(900, 700),
):
    """
    Build Plotly interactive wordcloud.

    Parameters
    ----------
    freq_dict : dict
        Mapping {word: frequency}.
    highlight_groups : dict
        {'GroupName': {'color': '#hex', 'words': [...]}}.
    """
    if not freq_dict:
        raise ValueError("Empty frequency dictionary.")
    set_style(style_mode)

    # Limit
    items = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:exact_words]
    words, freqs = zip(*items)
    max_f, min_f = max(freqs), min(freqs)
    sizes = [15 + (f / max_f) * 45 for f in freqs]

    # Improved placement with better collision detection
    rng = np.random.default_rng(42)
    positions, boxes = [], []
    for w, f, sz in zip(words, freqs, sizes):
        # Use frequency-based radius with more spacing
        rad = r_min + epsilon + (1 - f / max_f) * (spacing_factor - r_min - epsilon) * 2.0
        pt = None
        for attempt in range(max_attempts):
            # Spiral outward for better distribution
            th = rng.uniform(0, 2 * np.pi)
            spiral_factor = 1 + (attempt / max_attempts) * 1.5
            x, y = rad * spiral_factor * np.cos(th), rad * spiral_factor * np.sin(th)

            # Larger bounding boxes with more padding
            padding = sz * 0.8
            bb = (
                x - sz * len(w) * 0.5 - padding,
                x + sz * len(w) * 0.5 + padding,
                y - sz * 0.8 - padding,
                y + sz * 0.8 + padding
            )

            # Check collision
            collision = False
            for p in boxes:
                if not (bb[1] < p[0] or bb[0] > p[1] or bb[3] < p[2] or bb[2] > p[3]):
                    collision = True
                    break

            if not collision:
                pt, boxes = (x, y), boxes + [bb]
                break

        if pt is None:
            # Fallback: place far from center
            th = rng.uniform(0, 2 * np.pi)
            pt = (rad * 2.0 * np.cos(th), rad * 2.0 * np.sin(th))
        positions.append(pt)

    # Color assignment using proper viridis
    cmap = matplotlib.colormaps[colormap]
    colors = []
    for w, f in zip(words, freqs):
        color = None
        if highlight_groups:
            for g, info in highlight_groups.items():
                if w.lower() in [s.lower() for s in info["words"]]:
                    color = info["color"]
                    break
        if not color:
            t = (f - min_f) / (max_f - min_f + 1e-9)
            if use_custom_freq_colors:
                def blend_hex(h1, h2, t):
                    h1, h2 = h1.lstrip("#"), h2.lstrip("#")
                    r1, g1, b1 = int(h1[0:2], 16), int(h1[2:4], 16), int(h1[4:6], 16)
                    r2, g2, b2 = int(h2[0:2], 16), int(h2[2:4], 16), int(h2[4:6], 16)
                    return f"rgb({int(r1+(r2-r1)*t)},{int(g1+(g2-g1)*t)},{int(b1+(b2-b1)*t)})"
                color = blend_hex(low_freq_color, high_freq_color, t)
            elif use_gradient:
                r, g, b, _ = cmap(t)
                color = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
            else:
                # Default: use viridis gradient
                r, g, b, _ = cmap(t)
                color = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        colors.append(color)

    x, y = zip(*positions)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        text=[w.replace("_", " ") for w in words],
        mode="text",
        textfont=dict(size=sizes, color=colors),
        hovertext=[f"{w}: {f}" for w, f in zip(words, freqs)],
        hoverinfo="text"
    ))

    if highlight_groups:
        for g, info in highlight_groups.items():
            fig.add_trace(go.Scatter(x=[None], y=[None],
                                     mode='markers',
                                     marker=dict(size=10, color=info["color"]),
                                     name=g))

    fig.update_layout(
        title=title or "Interactive Wordcloud",
        showlegend=True,
        width=figsize[0],
        height=figsize[1],
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        hovermode="closest",
        margin=dict(t=60, b=40, l=40, r=40),
        legend_title_text="Highlight Groups"
    )
    return fig
