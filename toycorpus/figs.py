from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np

from toycorpus import configs


def make_info_theory_fig(ys: np.array,  # 2D: each row is a curve
                         title: str,
                         x_axis_label: str,
                         y_axis_label: str,
                         x_ticks: List[int],
                         labels: List[str],
                         y_lims: Optional[List[float]] = None,
                         ):
    fig, ax = plt.subplots(1, figsize=(6, 4), dpi=163)
    plt.title(title, fontsize=configs.Figs.title_font_size)
    ax.set_ylabel(y_axis_label, fontsize=configs.Figs.ax_font_size)
    ax.set_xlabel(x_axis_label, fontsize=configs.Figs.ax_font_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(['' if n + 1 != len(x_ticks) else f'{i:,}' for n, i in enumerate(x_ticks)],
                       fontsize=configs.Figs.tick_font_size)
    if y_lims:
        ax.set_ylim(y_lims)

    # plot
    lines = []  # will have 1 list for each condition
    for n, y in enumerate(ys):
        print(y)
        print(x_ticks)
        line, = ax.plot(x_ticks, y, linewidth=2, color=f'C{n}')
        lines.append([line])

    # legend
    plt.legend([l[0] for l in lines],
               labels,
               loc='upper center',
               bbox_to_anchor=(0.5, -0.3),
               ncol=2,
               frameon=False,
               fontsize=configs.Figs.leg_font_size)

    return fig, ax
