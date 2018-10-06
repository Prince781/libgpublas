#!/bin/env python3
# This script produces a graph of the data flow among functions in an ltrace output

import sys
import re
import gzip

class Node:
    def __init__(self, name, inputs, outputs):
        self.name = name        # function/symbol name
        self.inputs = inputs    # list of pointers/arguments going in
        self.outputs = outputs   # pointer coming out

    # combine two nodes with the same name
    def __add__(self, other_node):
        assert other_node.name == self.name
        return Node(self.name, self.inputs.union(other_node.inputs), self.outputs.union(other_node.outputs))

def parse_input(filename):
    try:
        inp = gzip.open(filename, 'rt') if filename != '-' else sys.stdin
    except IOError as err:
        print(err)
        return

    oup = sys.stdout
    pointer_re = r'0x[A-Za-z0-9]+'
    symbol_re = r'[A-Za-z]\w+'

    atomic_re = re.compile(fr'({symbol_re})\((.*)\)\s+=\s+(.*)')
    unfinished_re = re.compile(fr'({symbol_re})\((.*) <unfinished ...>')
    resumed_re = re.compile(fr'<... ({symbol_re}) resumed>\s+=\s+(.*)')

    nodes = {}      # [sym] = node
    outgoing = {}   # [ptr] = [node] means node outputs ptr

    unfinished = {} # when we see an 'end', we combine it with this and put it in nodes

    oup.write('digraph ltrace {\n')

    for line in inp:
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
            ret = match.group(3)

            inputs = set(re.findall(pointer_re, args_str))
            outputs = set(re.findall(pointer_re, ret))

            if not sym in nodes:
                nodes[sym] = Node(sym, inputs, outputs)
            else:
                nodes[sym] += Node(sym, inputs, outputs)

            for ptr in outputs:
                if not ptr in outgoing:
                    outgoing[ptr] = {}
                outgoing[ptr].update({sym: nodes[sym]})
                    
        elif mtype == 'begin':
            sym = match.group(1)
            args_str = match.group(2)
            inputs = set(re.findall(pointer_re, args_str))

            if not sym in unfinished:
                unfinished[sym] = []
            unfinished[sym].append(Node(sym, inputs, set()))

        else:   # 'end'
            sym = match.group(1)
            ret = match.group(2)

            if not sym in unfinished or not unfinished[sym]:
                raise Exception(f'`{line}` comes before a corresponding "unfinished"')

            outputs = set(re.findall(pointer_re, ret))

            node = unfinished[sym].pop()
            if not sym in nodes:
                nodes[sym] = node + Node(sym, set(), outputs)
            else:
                nodes[sym] += node + Node(sym, set(), outputs)

            for ptr in outputs:
                if not ptr in outgoing:
                    outgoing[ptr] = {}
                outgoing[ptr].update({sym: nodes[sym]})

    for sym, node in nodes.items():
        oup.write(f'\t{sym};\n')

        for in_ptr in node.inputs:
            if in_ptr in outgoing:
                for psym, parent in outgoing[in_ptr].items():
                    oup.write(f'\t{psym} -> {sym} [label="{in_ptr}"];\n')

    oup.write('}\n')
    inp.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} ltrace-file")
        sys.exit(1)

    parse_input(sys.argv[1])
