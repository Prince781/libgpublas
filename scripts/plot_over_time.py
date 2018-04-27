#!/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

if len(sys.argv) < 2:
    sys.exit(f'Usage: {sys.argv[0]} <events CSV file>')

fname = sys.argv[1]

# Data for plotting
df = pd.read_csv(fname, sep=r',\s*')

t = df['Time (ns)'] / 10**9

fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()


ax0.plot(t, df['Host Items'])
ax0.set(xlabel='time (s)', ylabel='host items', title='host items')
ax0.grid()

ax1.plot(t, df['Host Size (B)'])
ax1.set(xlabel='time (s)', ylabel='host size (B)', title='host memory')
ax1.grid()

ax2.plot(t, df['Dev Items'])
ax2.set(xlabel='time (s)', ylabel='device items', title='device items')
ax2.grid()

ax3.plot(t, df['Dev Size (B)'])
ax3.set(xlabel='time (s)', ylabel='device size (B)', title='device memory')
ax3.grid()

fig.tight_layout()
fig.savefig("test.svg")
plt.show()
