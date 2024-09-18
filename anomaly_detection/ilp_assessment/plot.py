import pandas as pd
import matplotlib.ticker as ticker
from numpy import *
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import os
import numpy as np

directories = os.listdir('./time')
time = {}
for dir in directories:
    time[dir] = {}
    files = os.listdir(f"./time/{dir}")
    for i,file in enumerate(files):
        df = pd.read_csv(f"./time/{dir}/{file}", delimiter=',', header=None)

        time[dir] = df.iloc[0:2, 1].to_numpy()


fig = plt.figure(figsize=(6,3), tight_layout=True)
x = list(range(1,8))
ax=plt.subplot(111)
plt.setp(ax.get_xticklabels(), fontweight='bold')
plt.setp(ax.get_yticklabels(), fontweight='bold')
new_labels_outer = ['50','33','25','20','16.6','14.2','12.5']
y_values = []
y_err = []
keys = list(time.keys())
reversed_keys = keys[::-1]
for key in reversed_keys:
    y_values.append(time[key][0])
    y_err.append(time[key][1])
ax.errorbar(x, y_values, yerr=y_err, color="dimgray", marker= '^',linestyle='-', ecolor='black', capsize=5)


ax.xaxis.set_major_locator(ticker.FixedLocator(x))
ax.yaxis.set_major_locator(MaxNLocator(nbins=15))
ax.grid(True, axis='y', color='grey', linestyle='-', linewidth=0.2, alpha=0.5)
ax.xaxis.set_major_formatter(ticker.FixedFormatter(new_labels_outer))
ax.set_xticklabels(new_labels_outer)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
keys = list(time.keys())
plt.grid(axis = 'y', color = 'grey', linestyle = '-', linewidth = 0.2)
plt.setp(ax.get_xticklabels(), fontweight='bold', fontsize = 9)
plt.setp(ax.get_yticklabels(), fontweight='bold', fontsize = 9)
plt.xlabel('Q [%]', fontweight = "bold", fontsize = 9)
plt.ylabel('Time [s]', fontweight = "bold", fontsize = 9)
plt.savefig('q_25.png',format="png",dpi=600, bbox_inches='tight')


files = os.listdir(f"./results")
perf = {}
for file in files:
    df = pd.read_csv(f"./results/{file}", delimiter=',', header=None)
    list_algorithm = df[0][:].to_numpy().tolist()
    perf[file] = df.iloc[:, 4].to_numpy()


colors = []

perf_ = {}

for i, alg in enumerate(list_algorithm):
    perf_[alg] = []
    list_dir = []
    for dir in perf.keys():
        list_dir.append(dir)
        perf_[alg].append(perf[dir][i])


print(perf_)
fig = plt.figure(figsize=(6,3), tight_layout=True)
width = 0.06
ax=plt.subplot(111)
x = np.arange(1,len(perf_.keys())+1)
new_labels_inner = list(perf_.keys())
new_labels_outer = ['50 %','33 %','25 %','20 %','16.6 %','14.2 %','12.5 %']
colors = ['dimgray', 'gray', 'darkgray', 'silver', 'lightgrey', 'whitesmoke', 'snow']
hatch_patterns = ['/'*5, '\\'*5, '|'*5, '-'*5, '+'*5, 'x'*5, 'o'*5]
for i, file in enumerate(perf_.keys()):
        ax.bar(x[i] - 6 * width, perf_[file][6]*100,width=width, hatch=hatch_patterns[0], color=colors[0], edgecolor='black', linewidth=0.5)
        ax.bar(x[i] - 4 * width, perf_[file][5]*100, width=width, hatch=hatch_patterns[1], color=colors[1], edgecolor='black', linewidth=0.5)
        ax.bar(x[i] - 2 * width, perf_[file][4]*100, width=width, hatch=hatch_patterns[2], color=colors[2], edgecolor='black', linewidth=0.5)
        ax.bar(x[i] - 0 * width, perf_[file][3]*100, width=width, hatch=hatch_patterns[3], color=colors[3], edgecolor='black', linewidth=0.5)
        ax.bar(x[i] + 2 * width, perf_[file][2]*100, width=width, hatch=hatch_patterns[4], color=colors[4], edgecolor='black', linewidth=0.5)
        ax.bar(x[i] + 4 * width, perf_[file][1]*100, width=width, hatch=hatch_patterns[5], color=colors[5], edgecolor='black', linewidth=0.5)
        ax.bar(x[i] + 6 * width, perf_[file][0]*100, width=width, hatch=hatch_patterns[6], color=colors[6], edgecolor='black', linewidth=0.5)

ax.xaxis.set_major_locator(ticker.FixedLocator(x))
ax.yaxis.set_major_locator(MaxNLocator(nbins=15))
ax.grid(True, axis='y', color='grey', linestyle='-', linewidth=0.2, alpha=0.5)
ax.xaxis.set_major_formatter(ticker.FixedFormatter(new_labels_inner))
ax.set_xticklabels(new_labels_inner, fontsize = 9, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.setp(ax.get_yticklabels(), fontweight='bold', fontsize = 9)
plt.ylabel('F1 [%]', fontweight = "bold", fontsize = 9)
font_prop = FontProperties(weight='bold', size=6)
ax.legend(new_labels_outer, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=7, prop=font_prop)
plt.savefig('ilp.png',format="png",dpi=600, bbox_inches='tight')
