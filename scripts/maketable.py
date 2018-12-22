#!/usr/bin/env python3

import tarfile
import pandas as pd
import os
import re

dfs = []

m = {'results-atlas': 'ATLAS', 'results-b2c-false': 'blas2cuda ($h(q,r)=0$)', 'results-b2c-random': 'blas2cuda ($h(q,r)=P(X > 0.5)$)'}

tar = tarfile.open("results.tar.gz", "r:gz")
for member in tar.getmembers():
    f = tar.extractfile(member)
    if f:
        df = pd.read_csv(f)
        dfs.append({'name': m[os.path.splitext(member.name)[0]], 'df': df})

def scient(s):
    return re.sub(r'([\d.]+)e([+-])(\d+)', r'\1\\times 10^{\2\3}', s)

with open('results-table.tex', 'w') as tab:
    tab.write('\\begin{tabular}{' + ''.join(['c' for x in range(0,len(dfs)+1)]) + '}\n')
    tab.write('\\hline\\\\\n')
    tab.write('&' + '&'.join(map(lambda x: x['name'], dfs)) + '\\\\\n')
    tab.write('\\hline\\\\\n')
    for index, row in dfs[0]['df'].iterrows():
        tab.write(f"$n={int(row['Size'])}$")
        for i in range(0, len(dfs)):
            tab.write('& $' + str('%.3g' % dfs[i]['df']['Time'][index]) + '\pm ' + scient(str('%.3g' % float(dfs[i]['df']['Stdev'][index]))) + '$')
        tab.write('\\\\\n')
    tab.write('\\end{tabular}\n')
