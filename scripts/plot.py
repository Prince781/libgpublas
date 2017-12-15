#!/bin/python

# see https://nbviewer.ipython.org/urls/bitbucket.org/hrojas/learn-pandas/raw/master/lessons/01%20-%20Lesson.ipynb

from pandas import DataFrame, read_csv

import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import matplotlib 
import math

if (len(sys.argv) != 2):
    print ("Usage:", sys.argv[0], "<csv>")
    sys.exit(1)

Location = sys.argv[1]
Columns = ['Matrix Size', 'Time(s)']

df = pd.read_csv(Location, names=Columns)

print (df)

Ns = df[Columns[0]]
Times = df[Columns[1]]
TimeDataSet = list(zip(Ns, Times))

print (df.dtypes)
TimedDataFrame = pd.DataFrame(data = TimeDataSet,
        columns=[Columns[0],Columns[1]]).sort_values([Columns[0]])
TimedDataFrame.set_index(Columns[0])
tdPlot = TimedDataFrame[Columns[1]].plot(title='Time per each operation',
        xticks=[math.log(i,2) for i in df[Columns[0]]])
tdPlot.set_xticklabels(df[Columns[0]])
tdPlot.set_xlabel(Columns[0])
tdPlot.set_ylabel(Columns[1])
tdFig = tdPlot.get_figure()
tdFig.savefig(os.path.basename(Location) + '-td.png')

#GFLOPSDataFrame = pd.DataFrame(data = list(zip(df['Matrix Size'], df['GFLOPS'])),
#        columns=[Columns[0],Columns[2]]).sort_values([Columns[0]])

#print GFLOPSDataFrame
#gdPlot = GFLOPSDataFrame[Columns[2]].plot(title='GFLOPS per each operation')
#gdPlot.set_xlabel(Columns[0])
#gdPlot.set_ylabel(Columns[2])
#gdFig = gdPlot.get_figure()
#gdFig.savefig(os.path.basename(Location) + '-gd.png')
