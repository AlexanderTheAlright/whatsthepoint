# Checking a figure

Two checks, in this order. Neither replaces the other.

## 1. The programmatic check

```python
from whatsthepoint.utils.layout import assert_clean, check_layout

assert_clean(fig)            # raises, listing every fault
faults = check_layout(fig)   # or get the list
```

It reads the bounding boxes of every text, legend and data mark from the
renderer and reports:

| Fault | Meaning |
|---|---|
| text over text | two tick labels, axis labels, titles, value labels or legend entries overlap |
| legend over data | a marker, line or bar lies under the legend box |
| text outside figure | a text runs off the canvas |
| mark outside axes | a marker, line end, arrow or bar crosses the axes edge |
| label outside axes | an in-axes label leaves the axes |
| mark under label | a marker, line or edge lies under a label or node box |
| label over bar | a value label sits on a bar |

Fix the figure, not the check: more room (figure size, `finish`,
`pad_limits`), fewer or shorter labels, broken lines. Do not save a figure
that fails.

## 2. The look

Open the PNG (never the SVG) and go region by region. Say what you see in each
before judging it.

| Region | What to check |
|---|---|
| Top | no figure title; adequate white space; top labels inside the axes |
| Left margin | y-axis label present, readable, reader-facing; tick labels legible and apart |
| Plot area | the finding is visible in under ten seconds; no mark covers another's label; every marker, bar end and interval cap is whole |
| Annotations | no label on a mark; consistent size; evenly placed |
| Bottom | x-axis label present; tick labels apart; the legend below the panels, boxed, bold, clear of the axis label |
| Right margin | end labels and colour bar label inside the canvas; no clutter |
| Empty space | no large unused area; the figure is sized to its content |

Then look again at about 800 px wide, the size a README or a page shows it:
text must still be readable and marks distinct.

Checklist:

- **A. Overlap**: no text on text or on data.
- **B. Legibility** at publication size (3.5 in or 7 in wide).
- **C. Labels**: descriptive wording, no variable codes.
- **D. Data accuracy**: compare at least three values on the figure with the
  source file. Any mismatch fails the whole figure.
- **E. Narrative**: the figure tells its story without explanation.
- **G. Design**: works in black and white; no needless gridlines; legend
  boxed, bold and opaque.
- **H. No tool residue**: no prompt text, placeholder wording or generator
  names in labels, in the SVG source or in image metadata.

Every region passes or fails; there is no "mostly fine". If a label can only
be fixed by removing it, say what information is lost.
