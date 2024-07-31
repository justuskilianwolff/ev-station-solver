import os

import matplotlib.pyplot as plt
from cycler import cycler


def set_params(use_tex=False, grey_scale=False, color="black"):
    """
    Sets matplotlib parameters so that all plots use the same style.
    Args:
        use_tex: whether to use tex or not
        grey_scale: use grey scale
        color: what color to use. Standard is black, for black backgrounds, white can be used.

    Returns:
        None
    """
    plt.rcParams.update(
        {
            "lines.linewidth": 2,
            "font.sans-serif": "Lucida Grande",
            "font.size": 35,
            # Axes and figure colors
            "axes.edgecolor": color,
            "axes.titlecolor": color,
            "axes.labelcolor": color,  # Color of x and y label
            "figure.facecolor": (0.0, 0.0, 0.0, 0.0),
            "axes.facecolor": (0.0, 0.0, 0.0, 0.0),
            # Spines
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Legend
            "legend.framealpha": 0,
            # Ticks (do I actually need them?)
            "xtick.color": color,
            "ytick.color": color,
            "xtick.labelcolor": color,
            "ytick.labelcolor": color,
            "grid.color": "gray",
            # Figsize and resolution
            "figure.dpi": 100,
            "savefig.dpi": 400,
            # "figure.figsize": [20, 10],
            "savefig.bbox": "tight",
        }
    )

    if use_tex:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.size": 40,
            }
        )

    if grey_scale:
        plt.rcParams["axes.prop_cycle"] = cycler(
            linestyle=[
                "-",
                (0, (5, 10)),
                "-.",
                ":",
                "-.",
            ]
        )
        plt.rcParams["lines.color"] = color
    else:
        pass


def save_fig(fig, name, chapter=None, path="../../media"):
    """
    Saves the figure with the specified chapter and path to that chapter.
    Args:
        fig: figure
        name: name of the plot
        chapter: what chapter?
        path: path to the plots fodler

    Returns:
        None
    """
    dirname = os.path.dirname(__file__)

    path_to_plots = os.path.join(dirname, path)

    if not chapter:
        path = os.path.join(path_to_plots, f"{name}.pdf")
    else:
        path = os.path.join(path_to_plots, chapter, f"{name}.pdf")

    fig.savefig(path)


def plot_arrows(ax, set_zero=True):
    """
    Used to plot arrows in the plots.
    Args:
        ax: axis
        set_zero: whether to center the plot to (0,0)

    Returns:

    """
    if set_zero:
        ax.spines[["left", "bottom"]].set_position("zero")
        ax.plot(
            1,
            0,
            ">",
            transform=ax.get_yaxis_transform(),
            clip_on=False,
            alpha=1,
            zorder=-1,
        )
        ax.plot(
            0,
            1,
            "^",
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            alpha=1,
            zorder=-1,
        )

    else:
        ax.plot(ax.get_xlim()[0], ax.get_ylim()[1], "^", clip_on=False)
        ax.plot(ax.get_xlim()[1], ax.get_ylim()[0], ">", clip_on=False)


def set_lims(ax, x_b, x_t, y_b, y_t, shift=0):
    """
    Set limits.
    Args:
        ax: ax
        x_b: x bottom
        x_t: x top
        y_b: y bottom
        y_t: y top
        shift: how much should be added?

    Returns:
        None
    """
    ax.set_xlim(x_b - shift, x_t + shift)
    ax.set_ylim(y_b - shift, y_t + shift)


def set_label(ax, x="x in km", y="y in km", tex=True):
    """

    Args:
        ax: axis
        x: x data
        y: y data
        z: z data
        tex: whether tex is used or not
        labelpad: how far away to position the axis

    Returns:

    """
    # if tex:
    #    x = r'\textbf{' + x + '}'
    #    y = r'\textbf{' + y + '}'

    ax.set_xlabel(x)
    ax.set_ylabel(y)
