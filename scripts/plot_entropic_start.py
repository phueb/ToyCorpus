"""
A new measure for quantifying whether a corpus "starts entropic":

ce_xy2 - ce_xy1, where
ce_xy1 measures the uncertainty about all words in corpus given their neighbors
ce_xy2 measures the uncertainty about individual target words given their neighbors

an "entropic start" should have a larger difference (e.g high ce_xy2, keeping constant ce_xy1)
in the un-reversed direction compared to the reversed direction.

"""

import numpy as np
from multiprocessing import Pool
from typing import List, Tuple
from pyitlib import discrete_random_variable as drv
import matplotlib.pyplot as plt

from preppy import Prep

from toycorpus.corpus import ToyCorpusTypes, ToyCorpusEntropic
from toycorpus import configs
from toycorpus.figs import make_info_theory_fig


# note: use CORPUS_NAME = 'toy-increase-others', or 'toy-decrease-entropy'

CORPUS_NAME = 'toy-decrease-entropy'
# CORPUS_NAME = 'toy-increase-others'

REORDER_NATURALISTIC = True
DISTANCE = + 1


def collect_data(ws: np.ndarray) -> Tuple[float, float]:

    ###############
    # val1: use all windows; this provides a control/reference from which to measure difference to val2
    # (should be 0.0 when using toy corpus)
    ###############

    # val1
    x1 = ws[:, -2]  # all words
    y1 = ws[:, -2 + DISTANCE]  # neighbors
    val1i = drv.entropy_conditional(x1, y1).item() / drv.entropy(x1).item()

    ###############
    # val2: use only target windows
    # in theory, this should be invariant to number of target types,
    ###############

    # target windows
    row_ids = np.isin(ws[:, -2], target_ids)
    target_windows = ws[row_ids]

    # val2
    x2 = target_windows[:, -2]  # target
    y2 = target_windows[:, -2 + DISTANCE]  # neighbors
    val2i = drv.entropy_conditional(x2, y2).item() / drv.entropy(x2).item()

    print(f'{len(ws):>12,} | val1={val1i:.3f} val2={val2i:.2f}')

    return val1i, val2i


if 'increase-nouns' in CORPUS_NAME:
    tc = ToyCorpusTypes(increase_noun_types=True, increase_other_types=False)
elif 'increase-others' in CORPUS_NAME:
    tc = ToyCorpusTypes(increase_noun_types=False, increase_other_types=True)
elif 'decrease-entropy' in CORPUS_NAME:
    tc = ToyCorpusEntropic()
else:
    raise AttributeError('Invalid arg to CORPUS_NAME')

prep = Prep(tc.tokens,
            reverse=False,
            sliding=False,
            num_parts=2,
            context_size=1)
targets = [p for p in tc.nouns if p in prep.token2id]


targets = set(targets)
target_ids = [prep.token2id[w] for w in targets]
windows = prep.reordered_windows

# collect results in parallel
pool = Pool(configs.Constants.num_processes)
val1 = [[], []]
val2 = [[], []]
x_ticks = [int(i) for i in np.linspace(0, len(windows), configs.Constants.num_ticks + 1)][1:]
for n, windows in enumerate([windows, np.flip(windows, 0)]):
    print()
    for val1i, val2i in pool.map(collect_data, [windows[:num_windows] for num_windows in x_ticks]):
        val1[n].append(val1i)
        val2[n].append(val2i)
pool.close()

title_additional = f'toy data with {tc.num_sentences:,} sentences\n' \
                   f'{CORPUS_NAME}\n'
try:
    title_additional += f'low-entropy probability range={tc.le_probabilities[0]} to {tc.le_probabilities[-1]}'
except AttributeError:
    pass


# fig
title = 'H(target X|Y) / H(target X) - H(all X|Y) / H(all X)' + '\n' + title_additional
x_axis_label = f'Location in {CORPUS_NAME} [num tokens]'
y_axis_label = 'Normalized Entropy'
labels = ['reversed=False', 'reversed=True']
make_info_theory_fig(
    ys=np.subtract(val2, val1),
    title=title,
    x_axis_label=x_axis_label,
    y_axis_label=y_axis_label,
    x_ticks=x_ticks,
    labels=labels,
    y_lims=[0.00, 0.20],
)
plt.show()
