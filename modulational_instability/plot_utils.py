"""Shared plotting helpers for the MI growth figures."""
import matplotlib.pyplot as plt

# Fixed geometry shared by both growth figures, in inches.  Saving with
# bbox_inches='tight' crops each figure to its own content, so the page size
# and the position of the axes would depend on what happens to be drawn.
AX_W, AX_H = 4.65, 5.2                     # axes size, same as before
LEFT, RIGHT, BOTTOM, TOP = 0.95, 0.2, 0.65, 0.15


def growth_figure():
    """Figure and axes with the fixed geometry above.  Save WITHOUT bbox_inches."""
    W = LEFT + AX_W + RIGHT
    H = BOTTOM + AX_H + TOP
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([LEFT / W, BOTTOM / H, AX_W / W, AX_H / H])
    return fig, ax


def keep_texts_inside(ax, pad=3):
    """Shift every ax.text label that sticks out of the axes back inside.

    Call this just before savefig, after the axis limits are set.
    `pad` is in pixels.
    """
    fig = ax.figure
    fig.canvas.draw()
    ax_bb = ax.get_window_extent()
    to_data = ax.transData.inverted()
    for txt in ax.texts:
        bb = txt.get_window_extent()
        dx = max(ax_bb.x0 + pad - bb.x0, 0) - max(bb.x1 - (ax_bb.x1 - pad), 0)
        dy = max(ax_bb.y0 + pad - bb.y0, 0) - max(bb.y1 - (ax_bb.y1 - pad), 0)
        if dx or dy:
            print(f"moved label {txt.get_text()!r} by ({dx:.0f}, {dy:.0f}) px")
            x, y = ax.transData.transform(txt.get_position())
            txt.set_position(to_data.transform((x + dx, y + dy)))