#!/bin/python

# see https://nbviewer.ipython.org/urls/bitbucket.org/hrojas/learn-pandas/raw/master/lessons/01%20-%20Lesson.ipynb

from pandas import DataFrame, read_csv

import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import matplotlib 
import math
import numpy as np

from matplotlib.ticker import NullFormatter

if (len(sys.argv) < 4):
    print ("Usage:", sys.argv[0], "<title> <CSV1 NAME1>")
    sys.exit(1)

Title = sys.argv[1]
DFs = []

# data to plot
fig, ax = plt.subplots()
index = []
bar_width = 0.2
domain = []

# taken from here: https://matplotlib.org/examples/api/barchart_demo.html
def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')


for i in range(2, len(sys.argv), 2):
    Location = sys.argv[i]
    Columns = ['Dimension of Matrix', 'Time (s)']

    df = pd.read_csv(Location, names=Columns)

    # print (df)
    DFs.append(df)

    Ns = df[Columns[0]]
    Times = df[Columns[1]]

    if len(Ns) > len(domain):
        domain = Ns
    index = np.arange(len(Ns))

    # create a new bar plot for this
    rects_i = plt.bar(index + (i/2 - 1) * bar_width, Times, bar_width, label=sys.argv[i + 1])

    #autolabel(rects_i)

    #TimeDataSet = list(zip(Ns, Times))

    # print (df.dtypes)
    #TimedDataFrame = pd.DataFrame(data = TimeDataSet,
    #        columns=[Columns[0],Columns[1]]).sort_values([Columns[0]])
    #TimedDataFrame.set_index(Columns[0])

    #tdPlot = TimedDataFrame[Columns[1]].plot(title='Time per call',
    #        xticks=[math.log(i,2) for i in df[Columns[0]]])
    #tdPlot.set_xticklabels(df[Columns[0]])
    #tdPlot.set_xlabel(Columns[0])
    #tdPlot.set_ylabel(Columns[1])

    #tdFig = tdPlot.get_figure()

    #plt.subplot(2, 2, i / 2)
    #plt.scatter(df[Columns[0]], df[Columns[1]])
    #plt.subplots_adjust()

    #plt.legend([sys.argv[i + 1]])
    #plt.xlabel(Columns[0])
    #plt.ylabel(Columns[1])
    #plt.ylim(0)
    #plt.grid(True)

#    for k in range(0, len(df[Columns[0]])):
#        plt.annotate(Times[k], (Ns[k],Times[k]+0.1), fontsize=8)

    # Format the minor tick labels of the y-axis into empty strings with
    # `NullFormatter`, to avoid cumbering the axis with too many labels.
    #plt.gca().yaxis.set_minor_formatter(NullFormatter())
    # Adjust the subplot layout, because the logit one may take more space
    # than usual, due to y-tick labels like "1 - 10^{-3}"
    #plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
    #                    wspace=0.35)

#plt.legend([sys.argv[i] for i in range(3,len(sys.argv), 2)])
#plt.yticks(np.arange(min(AllTimes), max(AllTimes), 0.1))
#plt.xscale('log')
plt.xlabel(Columns[0])
plt.ylabel(Columns[1])
plt.xticks(index + bar_width, domain)
plt.legend()
plt.tight_layout()
plt.title(Title)
plt.tight_layout()
plt.savefig('output.svg', dpi=144*2, format='svg')
print ("Wrote to ", "output.svg")

#GFLOPSDataFrame = pd.DataFrame(data = list(zip(df['Matrix Size'], df['GFLOPS'])),
#        columns=[Columns[0],Columns[2]]).sort_values([Columns[0]])

#print GFLOPSDataFrame
#gdPlot = GFLOPSDataFrame[Columns[2]].plot(title='GFLOPS per each operation')
#gdPlot.set_xlabel(Columns[0])
#gdPlot.set_ylabel(Columns[2])
#gdFig = gdPlot.get_figure()
#gdFig.savefig(os.path.basename(Location) + '-gd.png')
