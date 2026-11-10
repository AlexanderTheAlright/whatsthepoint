# House style

`set_style("house")` applies all of this; the chart functions call it for you.
The constants live in `whatsthepoint/utils/style.py`.

## Type

| Element | Setting |
|---|---|
| Font | sans-serif: DejaVu Sans, then Arial, Helvetica, Liberation Sans. Looked up by name, never bundled; DejaVu Sans ships with matplotlib, so there is always a fallback |
| Base size | 10 pt |
| Axis labels | 11 pt or larger, bold, sentence case |
| Tick labels | 10 pt, bold |
| Annotations | 10 pt or larger, bold |
| Minimum | never below 9 pt: fix the layout instead of shrinking text |
| SVG text | kept as text (`svg.fonttype = 'none'`) so it stays editable |

## Axes

- No top or right spine. Black axis lines, 1.0 pt.
- White figure and axes background.
- No gridlines by default. If a grid is asked for: y-axis only, `#E0E0E0`,
  0.5 pt, faint.
- Colour bar labels read top to bottom: `rotation=270, labelpad=15`. Never a
  title on the colour bar.

## Groups in black and white

Fills (bars, boxes, areas), in order:

| Groups | Fill |
|---|---|
| 1 | black, or white with a black outline |
| 2 | black, and white with a black outline. Never grey for two groups: it does not separate them |
| 3 and 4 | add `#888888` hatched `//` and `#CCCCCC` hatched `..` |
| 5 or more | sampled from the `Greys` colour map |

Every bar, box and marker has a black outline (1.0 pt).

Lines and points: all black. Groups differ by line style (`-`, `--`, `:`,
`-.`), marker shape (circle, square, triangle, diamond) and marker fill
(black, white, `#888888`, `#CCCCCC`). Lines are 2.5 pt, markers 7 pt with a
1.5 pt edge. A fitted line that crosses points gets a white casing so it stays
readable.

Estimates with intervals: filled marker when p < .05, open marker otherwise,
one marker size, black interval bars, dashed black line at zero.

Box plots: white boxes with black outlines, black median, open diamond for the
mean, open circles for outliers. The groups are named on the axis.

## Legend

Every legend, on one panel or many:

- one legend for the figure, centred below the panels
  (`fig.legend(loc="outside lower center")` under constrained layout)
- opaque white box (`framealpha=1.0`), dark border (`edgecolor="0.15"`, 1.0 pt)
- bold title and bold labels, 10 pt or larger (title 11 pt)
- `handlelength` 2.5 or more, with room between handle, text and box edge
- long entries broken over lines

A borderless strip of thin grey text is a defect.

## Labels and numbers

- Reader-facing wording on every axis, tick and legend entry: "Share agreeing
  (%)", not a column name.
- Sentence case. No jokes, no headline phrasing.
- Tick labels longer than 15 characters are broken over lines at word
  boundaries; they are not rotated unless there are many.
- Numbers: integers without decimals, otherwise one or two decimals, thousands
  separators (`smart_format`).

## Saving

`save_figure(fig, stem)` writes `stem.svg` and `stem.png` (300 dpi, white
background, tight bounding box), with timestamps stripped so a re-render of the
same figure gives the same file.

## Other modes

`viridis` (colour), `reviewer3` (black and white, earlier layout), `sentiment`,
`fiery`, `plainjane`. Use them only when asked.
