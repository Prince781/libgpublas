#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os

def fixname(name):
    return os.path.splitext(os.path.basename(name))[0]

if len(sys.argv) < 2:
    sys.exit(f'Usage: {sys.argv[0]} <csv ...>')

fname = sys.argv[1]

fig, axes = plt.subplots(nrows=2)
ax0, ax1 = axes.flatten()

# Data for plotting
dfs = []

for i in range(1, len(sys.argv)):
    dfs.append(pd.read_csv(sys.argv[i]))

t = dfs[0]['Size']

for i in range(0, len(dfs)):
    ax0.plot(t, dfs[i]['Time'], label=fixname(sys.argv[i+1]))
    ax1.plot(t, dfs[i]['GFLOPs'], label=fixname(sys.argv[i+1]))

ax0.set(xlabel='size (n)', ylabel='time (s)', title='Execution Time - Matrix Multiplication', yscale='log')
ax0.grid()
ax0.legend()

ax1.set(xlabel='size (n)', ylabel='GFLOPs', title='GFLOPs - Matrix Multiplication')
ax1.grid()

ax1.legend()

fig.tight_layout()
fig.savefig("plot_results.svg")
plt.show()
