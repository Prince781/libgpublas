#!/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

if len(sys.argv) < 3:
    sys.exit(f'Usage: {sys.argv[0]} <events CSV file> <trace file>')

eventsfile = sys.argv[1]
tracefile = sys.argv[2]

# Data for plotting
events_df = pd.read_csv(eventsfile, sep=', ')
trace_df = pd.read_csv(tracefile, sep=', ')

events = {}

Xs = []
Ys = []
Cs = []

for idx, row in events_df.iterrows():
    total = row['Host Size (B)'] + row['Dev Size (B)']
    events[row['Time (ns)']] = row['Host Size (B)'] / total

for idx, row in trace_df.iterrows():
    Xs.append(events[row['Created (ns)']])
    Ys.append(row['Size (B)'])
    Cs.append('blue' if row['Host'] else 'green')

fig, ax = plt.subplots()

ax.scatter(Xs, Ys, c=Cs)
ax.set_xlabel(r'ratio $\frac{host}{host + dev}$')
ax.set_ylabel(r'size (B)')
ax.set_title('objects')

fig.tight_layout()
fig.savefig("test.svg")
plt.show()
