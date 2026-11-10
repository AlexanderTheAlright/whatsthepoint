---
name: visualize
description: >
  Make or fix a figure, chart, plot or path diagram with the whatsthepoint
  package in its default `house` style: "plot this", "make a figure", "graph
  this", "show me", "draw the model", "fix this figure", "the legend overlaps",
  "something is cut off". Produces black-and-white, publication-ready figures
  as PNG and SVG, checked for overlap and clipping before they are saved.
---

# visualize: figures in the house style

Each figure tells one clear finding and never misleads statistically. Figures
are drawn through `whatsthepoint` with `style_mode="house"` (the default), and
no figure is saved until `assert_clean(fig)` passes and you have looked at the
PNG yourself.

Details live in the references; this file is the map.

| Read | For |
|---|---|
| [references/style.md](references/style.md) | fonts, sizes, black/white/grey fills, hatches, line styles, legend, labels |
| [references/layout.md](references/layout.md) | the rules that make overlap impossible, and the package helpers that apply them |
| [references/qa.md](references/qa.md) | the programmatic check and the region-by-region look at the PNG |
| `docs/samples/` and `docs/sim_make_samples.py` | worked examples on simulated data |

## Steps for one figure

1. **Lock the story before any code.** Write down:
   1. the one finding, as a sentence a reader leaves with;
   2. the chart type that shows it most clearly;
   3. the comparison that makes the argument visible;
   4. what the reader sees first.

   A vague finding ("distribution of responses across categories") is
   sharpened ("the oldest group is twice as likely to agree") or the figure is
   dropped. A flat null or a pattern nobody would doubt is not a figure. If
   you cannot answer all four, ask.
2. **Write the labels in plain text first**: axis labels, category names,
   legend entries. Full reader-facing wording, sentence case, never a variable
   code.
3. **Draw it** with the chart function for the type (table below). One script
   per figure, named for the figure it writes; notes go in the script's header
   docstring (what it shows, data file, sample and N, construction choices,
   caveats, suggested caption).
4. **Check it**: `assert_clean(fig)` before saving. Fix the layout, never the
   check.
5. **Save both formats** with `save_figure(fig, "fig.01.slug")`: SVG is the
   vector source, PNG (300 dpi) is the raster and the copy you inspect.
6. **Look at the PNG**, region by region, at full size and at about 800 px
   wide ([references/qa.md](references/qa.md)). Never ship the first render
   unseen. Up to three fix cycles; after that, say what is still wrong and
   what fixing it would cost.
7. **Report**: the story sentence, chart type and why, the paths, what the
   check and the look found, which values you compared with the source.

## Chart functions

| Finding | Function |
|---|---|
| Quantities across categories | `visuals.bar.bar` (`orientation="vertical"`, `"horizontal"`, `"grouped"`, `"stacked"`) |
| Regression estimates with intervals | `visuals.coef.coef`, `visuals.oddsratio`, `visuals.margins` |
| Change over time | `visuals.trend.trend` |
| Shape of a distribution | `visuals.density.kde`, `visuals.boxplot.boxplot`, `visuals.hist` |
| Two variables | `visuals.scatter.scatterplot` |
| Several panels | matplotlib subplots after `set_style("house")`, then `utils.layout.finish(fig)` |
| Causal or path diagram | `visuals.dag.dag` |
| Terms that occur together | `visuals.cooccur.cooccur` |

`bar`, `coef`, `trend`, `kde`, `boxplot`, `scatterplot`, `dag` and `cooccur`
apply the whole house layout themselves. The other chart modules take the
house palette and fonts; finish them with `utils.layout.finish(fig)` and run
the check.

## Rules that always hold

- **Black and white by default.** Colour only when asked (`style_mode="viridis"`).
- **No figure title and no note line in the image.** Captions and notes belong
  in the manuscript, under the figure. Sub-panel titles ("Region one") are fine.
- **Never hardcode a statistic.** Every N, mean, estimate or p-value shown is
  read from the analysis output.
- **One legend per figure, below the panels**, boxed, opaque and bold. Never a
  legend per panel. Drop a legend that is not needed.
- **Nothing overlaps and nothing is clipped.** Every marker, bar end, interval
  cap, node and label is fully inside the axes; no label covers a mark.
- **Annotate sparingly**: only the values the finding needs, at most five per
  panel, evenly placed, bold, 10 pt or larger.
- **Value axes start at zero** for counts and proportions unless there is a
  stated reason. No cherry-picked ranges. Intervals wherever the data support
  them.
- **Results from one cross-section are associations.** Labels and legends say
  "association", never "effect".
- **Plain code.** One script tells the whole story; sparse lowercase comments;
  no helper layers around a single figure.
- **Ask when unsure** about data mapping, labels or variables.

## Tables

A table is data first: a CSV holding every estimate with its interval or
standard error, typeset afterwards. Stub labels are reader-facing wording,
never variable names. Notes state facts (sample, source, estimator,
significance key).
