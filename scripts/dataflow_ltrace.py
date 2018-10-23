#!/bin/env python3
# This script produces a graph of the data flow among functions in an ltrace output

import sys
import re
import gzip

num_nodes = 0
node_nums = []

arg_parsers = {\
        'sgemm_': [ {'ptr': 6, 'nrows': 2, 'ncols': 4}, {'ptr': 8, 'nrows': 4, 'ncols': 3}, {'ptr': 11, 'nrows': 2, 'ncols': 3, 'output': True} ],
        'cblas_sgemm': [ {'ptr': 7, 'nrows': 3, 'ncols': 5}, {'ptr': 9, 'nrows': 5, 'ncols': 4}, {'ptr': 12, 'nrows': 3, 'ncols': 4, 'output': True} ],
        'dgemm_': [ {'ptr': 6, 'nrows': 2, 'ncols': 4}, {'ptr': 8, 'nrows': 4, 'ncols': 3}, {'ptr': 11, 'nrows': 2, 'ncols': 3, 'output': True} ],
        'cblas_dgemm': [ {'ptr': 7, 'nrows': 3, 'ncols': 5}, {'ptr': 9, 'nrows': 5, 'ncols': 4}, {'ptr': 12, 'nrows': 3, 'ncols': 4, 'output': True} ],
        }

class Node:
    def __init__(self, name, inputs, outputs, optable):
        self.name = name        # function/symbol name
        self.inputs = inputs    # list of pointers/arguments going in
        self.outputs = outputs  # pointer coming out
        self.optable = optable  # in-flight operands
        global num_nodes
        num_nodes += 1
        self.id = num_nodes

    # combine two nodes with the same name
    def __add__(self, other_node):
        assert other_node.name == self.name
        return Node(self.name, self.inputs.union(other_node.inputs), \
                self.outputs.union(other_node.outputs), \
                other_node.optable)

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

    def contains(self, ptr):
        return ptr in self.ops

def print_node(oup, sym, node, outgoing):
    edges = []
    tbl = node.optable.ops

    for in_ptr in node.inputs:
        nrows = tbl[in_ptr]['nrows']# if in_ptr in tbl else 0
        ncols = tbl[in_ptr]['ncols']# if in_ptr in tbl else 0
        if in_ptr in outgoing:
            parent = outgoing[in_ptr]
            psym = parent.name
            if parent == node:
                continue
            dims = ''
            if nrows != 0 or ncols != 0:
                dims = f'[{nrows}x{ncols}]'
            edges.append(f'\t{psym}_{parent.id} -> {sym}_{node.id} [label="{in_ptr}{dims}"];\n')

    oup.write(f'\t{{ rank=same; "{node.id}"; {sym}_{node.id} [label="{sym}", shape="box", font="monospace"]; }};\n')
    oup.writelines(edges)
    node_nums.append(node.id)

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
    outgoing = {}   # [ptr] = node means node outputs ptr
    unfinished = {} # when we see an 'end', we combine it with this and put it in nodes

    # 1. keep optable for each node
    # 2. update optable 
    optable = OperandTable()    # in-flight operands

    oup.write('digraph thread1 {\n')
    oup.write('\tnode [shape=plaintext, fontsize=16];\n')

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

            inputs = set()
            args = [x.strip() for x in args_str.split(',')]
            outputs = set(re.findall(pointer_re, ret))

            if sym in arg_parsers:
                for arg_desc in arg_parsers[sym]:
                    arg = {key: args[idx] for key,idx in arg_desc.items()}
                    arg['nrows'] = int(arg['nrows'])
                    arg['ncols'] = int(arg['ncols'])
                    inputs.add(arg['ptr'])
                    optable += arg
                    if 'output' in arg and arg['output']:
                        outputs.add(arg['ptr'])
            else:
                inputs = set(re.findall(pointer_re, args_str))
                for x in inputs:
                    if not optable.contains(x):
                        optable += {'ptr': x, 'nrows': 0, 'ncols': 0}


            nodes[sym] = Node(sym, inputs, outputs, optable)

            print_node(oup, sym, nodes[sym], outgoing)

            for ptr in outputs:
                outgoing[ptr] = nodes[sym]
                    
        elif mtype == 'begin':
            sym = match.group(1)
            args_str = match.group(2)
            inputs = set()

            args = [x.strip() for x in args_str.split(',')]
            outputs = set()

            if sym in arg_parsers:
                for arg_desc in arg_parsers[sym]:
                    arg = {key: args[idx] for key,idx in arg_desc.items()}
                    arg['nrows'] = int(arg['nrows'])
                    arg['ncols'] = int(arg['ncols'])
                    inputs.add(arg['ptr'])
                    optable += arg
                    if 'output' in arg and arg['output']:
                        outputs.add(arg['ptr'])
            else:
                inputs = set(re.findall(pointer_re, args_str))
                for x in inputs:
                    if not optable.contains(x):
                        optable += {'ptr': x, 'nrows': 0, 'ncols': 0}

            if not sym in unfinished:
                unfinished[sym] = []
            unfinished[sym].append(Node(sym, inputs, outputs, optable))

        else:   # 'end'
            sym = match.group(1)
            ret = match.group(2)

            if not sym in unfinished or not unfinished[sym]:
                raise Exception(f'`{line}` comes before a corresponding "unfinished"')

            outputs = set(re.findall(pointer_re, ret))

            node = unfinished[sym].pop()
            nodes[sym] = node + Node(sym, set(), outputs, optable)

            print_node(oup, sym, nodes[sym], outgoing)

            for ptr in outputs:
                outgoing[ptr] = nodes[sym]

    oup.write(f'\t{{')
    if len(node_nums) > 0:
        oup.write('\t\t')
    for i in range(0, len(node_nums)):
        if i == len(node_nums) - 1:
            oup.write(f'"{node_nums[i]}";\n')
        else:
            oup.write(f'"{node_nums[i]}" -> ')
    oup.write(f'\t}}\n')

    oup.write('}\n')
    inp.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} ltrace-file")
        sys.exit(1)

    parse_input(sys.argv[1])
