#!/bin/env python3
# This script produces a graph of the data flow among functions in an ltrace output

import sys
import re
import gzip
import binascii
from blas_api import arg_parsers

num_nodes = 0
node_nums = []  # needed for layout
live_ops = {}   # [ptr] -> [node]

# histogram stuff
want_hist = False
blas_hist = {}  # maps path length -> count
path_lens = {}  # maps ptr -> current path length
total_blas = 0

class Node:
    def __init__(self, sym, in_edges=None, is_blas_func=None):
        global want_hist
        global num_nodes
        global total_blas
        self.sym = sym
        self.in_edges = in_edges if in_edges != None else {}
        self.id = num_nodes
        num_nodes += 1
        self.is_blas_func = is_blas_func if is_blas_func != None else False

        if self.is_blas_func:
            total_blas += 1

        if self.is_blas_func and want_hist:
            for e,n in self.in_edges.items():
                if n.is_blas_func:
                    path_lens[e] = (path_lens[e]+1) if e in path_lens else 1
        elif not self.is_blas_func and want_hist:
            # check if we're no longer on a BLAS path;
            for e,n in self.in_edges.items():
                if not e in path_lens:  # could happen if only one BLAS call was made before
                    continue
                if n.is_blas_func:  # remove
                    if path_lens[e] in blas_hist:
                        blas_hist[path_lens[e]] += 1
                    else:
                        blas_hist[path_lens[e]] = 1
                    del path_lens[e]

    def print(self, oup):
        global want_hist
        oup.write(f'\t{{ rank=same; "{self.id}"; {self.sym}_{self.id} [label="{self.sym}", shape="box"]; }};\n')
        for in_ptr,parent in self.in_edges.items():
            oup.write(f'\t{parent.sym}_{parent.id} -> {self.sym}_{self.id} [label="{in_ptr}"];\n')
        node_nums.append(self.id)
        if not want_hist:
            self.in_edges = None    # break references to save memory

# see https://stackoverflow.com/questions/3703276/how-to-tell-if-a-file-is-gzip-compressed
def is_gz_file(filepath):
    with open(filepath, 'rb') as test_f:
        return binascii.hexlify(test_f.read(2)) == b'1f8b'


def parse_input(filename):
    try:
        if filename == '-':
            inp = sys.stdin
        else:
            if is_gz_file(filename):
                inp = gzip.open(filename, 'rt')
            else:
                inp = open(filename, 'rt')
    except IOError as err:
        print(err, file=sys.stderr)
        return

    oup = sys.stdout

    pointer_re = r'0x[A-Za-z0-9]+'
    symbol_re = r'[A-Za-z]\w+'

    atomic_re = re.compile(fr'({symbol_re})\((.*)\)\s+=\s+(.*)')
    unfinished_re = re.compile(fr'({symbol_re})\((.*) <unfinished ...>')
    resumed_re = re.compile(fr'<... ({symbol_re}) resumed>\s+=\s+(.*)')

    unfinished = {} # when we see an 'end', we combine it with this and put it in nodes

    if not want_hist:
        oup.write('digraph thread1 {\n')
        oup.write('\tnode [shape=plaintext, fontsize=16];\n')

    lineno = 0

    for line in inp:
        lineno += 1

        match = atomic_re.match(line)
        mtype = 'node'

        if not match:
            match = unfinished_re.match(line)
            mtype = 'begin'
        if not match:
            match = resumed_re.match(line)
            mtype = 'end'
        if not match:
            continue

        if mtype == 'node':
            sym = match.group(1)
            args_str = match.group(2)
            ret_str = match.group(3)

            args = [x.strip() for x in args_str.split(',')]
            rets = list(re.findall(pointer_re, ret_str))

            in_edges = {arg: live_ops[arg] for arg in args if arg in live_ops}

            if sym == 'free' and len(args) > 0 and args[0] in live_ops:
                del live_ops[args[0]]

            nd = Node(sym, in_edges, sym in arg_parsers)

            if not want_hist:
                nd.print(oup)

            if sym in arg_parsers:
                for argd in arg_parsers[sym]:
                    if argd['output']:
                        live_ops[args[argd['ptr']]] = nd

            for ret in rets:
                live_ops[ret] = nd

        elif mtype == 'begin':
            sym = match.group(1)
            args_str = match.group(2)

            args = [x.strip() for x in args_str.split(',')]
            rets = list(re.findall(pointer_re, ret_str))

            in_edges = {arg: live_ops[arg] for arg in args if arg in live_ops}

            nd = (sym, in_edges, sym in arg_parsers)
            if not sym in unfinished:
                unfinished[sym] = []
            unfinished[sym].append(nd)

            if sym in arg_parsers:
                for argd in arg_parsers[sym]:
                    if argd['output']:
                        live_ops[args[argd['ptr']]] = nd
        else: # 'end'
            sym = match.group(1)
            ret_str = match.group(2)

            rets = list(re.findall(pointer_re, ret_str))

            if sym == 'free' and len(args) > 0 and args[0] in live_ops:
                del live_ops[args[0]]

            nd_sym, nd_in_edges, nd_is_blas = unfinished[sym].pop()

            nd = Node(nd_sym, nd_in_edges, nd_is_blas)

            if not want_hist:
                nd.print(oup)

            for ret in rets:
                live_ops[ret] = nd

    # used for formatting
    if not want_hist:
        oup.write(f'\t{{')
        if len(node_nums) > 0:
            oup.write('\t\t')
        for i in range(0, len(node_nums)):
            if i == len(node_nums) - 1:
                oup.write(f'"{node_nums[i]}";\n')
            else:
                oup.write(f'"{node_nums[i]}" -> ')
        oup.write(f'\t}}\n')

        oup.write('}\n')    # close digraph description
    else: # print histogram
        oup.write('Path Length: Count (Percent of Total BLAS Calls)\n')
        total_paths = 0
        total_path_len = 0
        for plen in sorted(blas_hist):
            total_path_len += (plen+1)*blas_hist[plen]
            pcnt = (plen+1)*blas_hist[plen]/total_blas*100
            oup.write(f'{plen}: {blas_hist[plen]} ({float("%.3g" % pcnt)}%)\n')
            total_paths += blas_hist[plen]
        oup.write(f'Total BLAS calls: {total_blas}\n')
        oup.write(f'Total BLAS paths: {total_paths}\n')
        oup.write(f'Isolated BLAS calls: {total_blas - total_path_len}\n')

if __name__ == "__main__":
    import sys
    from argparse import *

    aparser = ArgumentParser()
    aparser.add_argument('ltrace_file')
    aparser.add_argument('-c', '--hist', action='store_true', help='Only print histogram of BLAS call chains')

    args = aparser.parse_args()
    want_hist = args.hist
    parse_input(args.ltrace_file)
