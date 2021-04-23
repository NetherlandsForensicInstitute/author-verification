from lir import Xy_to_Xn
import numpy as np
from matplotlib import pyplot as plt
import json
import os

# path_prefix = 'n_pairs_per_class'
# values = ['100', '2000', '4000']
# path_prefix = 'n_frequent_words'
# values = ['10', '200', '500']
path_prefix = 'tokens_per_sample'
values = ['200', '600', '1200']
all_xplots = {}
all_perc0 = {}
all_perc1 = {}
paper_notation = {'tokens_per_sample': 'l', 'n_frequent_words': 'f', 'n_pairs_per_class': 'n_{max}'}

for v in values:
    keyval = str(v)
    lrs_path = os.path.join('output', 'txt_files', f'{path_prefix}={v}.txt')
    with open(lrs_path) as f:
        data = json.load(f)

    lrs = np.array(data['lrs'])
    y = np.array(data['y'])
    log_lrs = np.log10(lrs)

    xplot = np.linspace(np.min(log_lrs), np.max(log_lrs), 100)
    all_xplots[keyval] = xplot
    lr_0, lr_1 = Xy_to_Xn(log_lrs, y)
    all_perc0[keyval] = (sum(i >= xplot for i in lr_0) / len(lr_0)) * 100
    all_perc1[keyval] = (sum(i >= xplot for i in lr_1) / len(lr_1)) * 100

plot_lines = []
line_types = [':', '-', '--', (0, (1, 10)), (0, (1, 1)), (0, (5, 10)), (0, (5, 1)), (0, (3, 10, 1, 10)),
              (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5, 1, 5)), (0, (3, 10, 1, 10, 1, 10))]

plt.rc('font', size=11.5)
for i in range(len(values)):
    lw = 1.4 if (line_types[i % len(line_types)] == '-') and (i < len(line_types)) else 1.1
    la, = plt.plot(all_xplots[values[i]], all_perc1[values[i]], color='b', ls=line_types[i % len(line_types)], lw=lw)
    lb, = plt.plot(all_xplots[values[i]], all_perc0[values[i]], color='m', ls=line_types[i % len(line_types)], lw=lw)
    plot_lines.append([la, lb])

legend1 = plt.legend(plot_lines[1], ['LRs given $\mathregular{H_{ss}}$', 'LRs given $\mathregular{H_{ds}}$'], loc=3)
text_legend2 = '$\mathregular{' + paper_notation[path_prefix] + '}$ = '
plt.legend([line[0] for line in plot_lines], [text_legend2 + s for s in values], loc=1)
plt.gca().add_artist(legend1)

plt.axvline(x=0, color='k', ls=(0, (5, 10)), lw=0.2, alpha=0.4)
plt.xlabel('Log likelihood ratio')
plt.ylabel('Cumulative proportion')

plt.savefig(os.path.join('output', 'paper_plots', f'{path_prefix}_tippet_plots_1.pdf'), bbox_inches='tight',
    pad_inches=0)
plt.savefig(os.path.join('output', 'paper_plots', f'{path_prefix}_tippet_plots.png'))
plt.close()
