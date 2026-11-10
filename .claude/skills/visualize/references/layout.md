# Layout: overlap prevented by construction

A model cannot reliably see overlap in a rendered figure, so the layout makes
it impossible first and measures second. The helpers are in
`whatsthepoint/utils/layout.py`.

## Structure

1. **One shared legend below the panels.** `finish(fig)` collects the entries
   from every axes, removes per-panel legends and places one boxed legend
   under the figure. A legend inside the plot area can cover data; one below
   it cannot.
2. **Constrained layout**, never `tight_layout` alone. Margins come from the
   measured extents of tick labels, axis labels, panel titles and the legend.
   Panels are spaced widely.
3. **Size generously.** One panel at least 5 in wide, two panels at least 9 in,
   three at least 10 in.
4. **Long labels are broken over lines** (`wrap_label`, 15 characters for
   ticks, 28 for legend entries) so they cannot run into a neighbour.
5. **Limits are padded until every mark is inside** (`pad_limits`): the axes
   grow until each marker, line end, interval cap, arrow and in-axes label is
   inside with a few points to spare. A zero baseline is kept by naming the
   sides that may move (`pad="t"` for vertical bars).

## Value labels

- Anchored to the data point and offset in points (`value_label`), never
  placed at raw data coordinates, so the gap to the mark is the same at any
  figure size.
- Bar labels go above the bar (beside it for horizontal bars), all at the same
  offset.
- Estimate labels go the same distance above every estimate, clear of the
  interval bar.
- End-of-line labels go to the right of the last point, never on it, and are
  spread evenly when two would touch (`spread`).
- If labels cannot fit side by side they lose their boxes first and are
  dropped only if they still collide. Never shrink them below 10 pt.
- Outside these patterns, use `adjustText` whenever two or more annotations
  could meet, then run the check.

## Node-link figures

- **A node is its own label box** (`dag`): the node cannot sit hidden under
  its label, and the box always fits the text.
- An edge that would run behind another node is bent until it clears it.
- An edge label sits on its own edge, clear of every node, every other label
  and every other edge.
- In a co-occurrence network (`cooccur`) the central term is in the middle and
  its neighbours are spaced evenly round it; each label sits beside its node,
  on the side away from the centre.

## Several panels

```python
import matplotlib.pyplot as plt
from whatsthepoint.utils.style import set_style
from whatsthepoint.utils.layout import finish, assert_clean

set_style("house")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
# ... draw on each axes, giving each series a label ...
finish(fig)          # one legend below both panels, limits padded
assert_clean(fig)
```
