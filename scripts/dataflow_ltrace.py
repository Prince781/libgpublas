#!/bin/env python3
# This script produces a graph of the data flow among functions in an ltrace output

import sys
import re
import gzip

arg_parsers = {\
        'sgemm_': [ {'ptr': 6, 'nrows': 2, 'ncols': 4}, {'ptr': 8, 'nrows': 4, 'ncols': 3}, {'ptr': 11, 'nrows': 2, 'ncols': 3} ],
        'dgemm_': [ {'ptr': 6, 'nrows': 2, 'ncols': 4}, {'ptr': 8, 'nrows': 4, 'ncols': 3}, {'ptr': 11, 'nrows': 2, 'ncols': 3} ],
        }

class Node:
    def __init__(self, name, inputs, outputs, optable):
        self.name = name        # function/symbol name
        self.inputs = inputs    # list of pointers/arguments going in
        self.outputs = outputs  # pointer coming out
        self.optable = optable  # in-flight operands

    # combine two nodes with the same name
    def __add__(self, other_node):
        assert other_node.name == self.name
        return Node(self.name, self.inputs.union(other_node.inputs), \
                self.outputs.union(other_node.outputs), \
                self.optable)

# immutable
class OperandTable:
    def __init__(self, ops=None):
        self.ops = {} if ops == None else ops

    def __add__(self, other):
        assert 'ptr' in other
        assert 'nrows' in other
        assert 'ncols' in other
        newop = {k:other[k] for k in other if k != 'ptr'}
        if other['ptr'] in self.ops:
            ops = self.ops.copy()
            ops[other['ptr']] = newop
            return OperandTable(ops)
        else:
            self.ops[other['ptr']] = newop
        return self

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

    # 1. keep optable for each node
    # 2. update optable 
    optable = OperandTable()    # in-flight operands

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

            inputs = [] # set(re.findall(pointer_re, args_str))
            args = args_str.split(',')

            if sym in arg_parsers:
                for arg_desc in arg_parsers[sym]:
                    arg = {key: args[idx] for key,idx in arg_desc.items()}
                    inputs.append(arg['ptr'])
                    optable += arg
            else:
                inputs = set(re.findall(pointer_re, args_str))
                for x in inputs:
                    optable += {'ptr': x, 'nrows': 0, 'ncols': 0}

            outputs = set(re.findall(pointer_re, ret))

            if not sym in nodes:
                nodes[sym] = Node(sym, inputs, outputs, optable)
            else:
                nodes[sym] += Node(sym, inputs, outputs, optable)

            for ptr in outputs:
                if not ptr in outgoing:
                    outgoing[ptr] = {}
                outgoing[ptr].update({sym: nodes[sym]})
                    
        elif mtype == 'begin':
            sym = match.group(1)
            args_str = match.group(2)
            inputs = []

            args = args_str.split(',')

            if sym in arg_parsers:
                for arg_desc in arg_parsers[sym]:
                    arg = {key: args[idx] for key,idx in arg_desc.items()}
                    inputs.append(arg['ptr'])
                    optable += arg
            else:
                inputs = set(re.findall(pointer_re, args_str))
#                for x in inputs:
#                    optable += {'ptr': x, 'nrows': 0, 'ncols': 0}

            if not sym in unfinished:
                unfinished[sym] = []
            unfinished[sym].append(Node(sym, inputs, set(), optable))

        else:   # 'end'
            sym = match.group(1)
            ret = match.group(2)

            if not sym in unfinished or not unfinished[sym]:
                raise Exception(f'`{line}` comes before a corresponding "unfinished"')

            outputs = set(re.findall(pointer_re, ret))

            node = unfinished[sym].pop()
            if not sym in nodes:
                nodes[sym] = node + Node(sym, set(), outputs, optable)
            else:
                nodes[sym] += node + Node(sym, set(), outputs, optable)

            for ptr in outputs:
                if not ptr in outgoing:
                    outgoing[ptr] = {}
                outgoing[ptr].update({sym: nodes[sym]})

    for sym, node in nodes.items():
        edges = []
        tbl = node.optable.ops

        for in_ptr in node.inputs:
            nrows = tbl[in_ptr]['nrows']
            ncols = tbl[in_ptr]['ncols']
            if in_ptr in outgoing:
                for psym, parent in outgoing[in_ptr].items():
                    edges.append(f'\t{psym} -> {sym} [label="{in_ptr}[{nrows}x{ncols}]"];\n')

        if not edges:
            continue

        oup.write(f'\t{sym};\n')
        oup.writelines(edges)

    oup.write('}\n')
    inp.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} ltrace-file")
        sys.exit(1)

    parse_input(sys.argv[1])
