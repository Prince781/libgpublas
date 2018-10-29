#!/bin/env python3
# This script produces a graph of the data flow among functions in an ltrace output

import sys
import re
import gzip
from blas_api import *

num_nodes = 0
node_nums = []

class Node:
    def __init__(self, name, inputs, outputs, optable, parents, on_blas_path):
        self.name = name        # function/symbol name
        self.inputs = inputs    # list of pointers/arguments going in
        self.outputs = outputs  # pointer coming out
        self.optable = optable  # in-flight operands
        global num_nodes
        num_nodes += 1
        self.id = num_nodes
        self.parents = parents
        self.on_blas_path = on_blas_path

    # combine two nodes with the same name
    def __add__(self, other_node):
        assert other_node.name == self.name
        return Node(self.name, self.inputs.union(other_node.inputs), \
                self.outputs.union(other_node.outputs), \
                other_node.optable, dict(list(self.parents.items()) + list(other_node.parents.items())), \
                self.on_blas_path or other_node.on_blas_path)

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

def print_node(oup, node):
    edges = []
    sym = node.name
    tbl = node.optable.ops

    for in_ptr in node.inputs:
        nrows = tbl[in_ptr]['nrows']# if in_ptr in tbl else 0
        ncols = tbl[in_ptr]['ncols']# if in_ptr in tbl else 0
        if in_ptr in node.parents:
            parent = node.parents[in_ptr]
            psym = parent.name
            if parent == node:
                continue
            dims = ''
            if nrows != 0 or ncols != 0:
                dims = f'[{nrows}x{ncols}]'
            edges.append(f'\t{psym}_{parent.id} -> {sym}_{node.id} [label="{in_ptr}{dims}"];\n')

    oup.write(f'\t{{ rank=same; "{node.id}"; {sym}_{node.id} [label="{sym}", shape="box"]; }};\n')
    oup.writelines(edges)
    node_nums.append(node.id)

def parse_input(filename, remove_non_blas=None):
    try:
        inp = gzip.open(filename, 'rt') if filename != '-' else sys.stdin
    except IOError as err:
        print(err)
        return

    remove_non_blas = False if remove_non_blas == None else remove_non_blas

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

    nodes_stack = []

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

            parents = {}

            on_blas_path = sym in arg_parsers

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

            for in_ptr in inputs:
                if in_ptr in outgoing:
                    parents.update({in_ptr: outgoing[in_ptr]})
                    on_blas_path |= outgoing[in_ptr].name in arg_parsers

            nodes[sym] = Node(sym, inputs, outputs, optable, parents, on_blas_path)

            nodes_stack.append(nodes[sym])

            for ptr in outputs:
                outgoing[ptr] = nodes[sym]
                    
        elif mtype == 'begin':
            sym = match.group(1)
            args_str = match.group(2)
            inputs = set()

            args = [x.strip() for x in args_str.split(',')]
            outputs = set()
            
            parents = {}

            on_blas_path = sym in arg_parsers

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

            for in_ptr in inputs:
                if in_ptr in outgoing:
                    parents.update({in_ptr: outgoing[in_ptr]})
                    on_blas_path |= outgoing[in_ptr].name in arg_parsers

            if not sym in unfinished:
                unfinished[sym] = []
            unfinished[sym].append(Node(sym, inputs, outputs, optable, parents, on_blas_path))

        else:   # 'end'
            sym = match.group(1)
            ret = match.group(2)

            if not sym in unfinished or not unfinished[sym]:
                raise Exception(f'`{line}` comes before a corresponding "unfinished"')

            outputs = set(re.findall(pointer_re, ret))

            node = unfinished[sym].pop()
            nodes[sym] = node + Node(sym, set(), outputs, optable, {})

            nodes_stack.append(nodes[sym])

            for ptr in outputs:
                outgoing[ptr] = nodes[sym]

    # now remove non-blas nodes
    final_nodes_stack = []

    if remove_non_blas:
        for node in reversed(nodes_stack):
            if not node.on_blas_path:
                continue
            for sym, parent in node.parents.items():
                parent.on_blas_path = node.on_blas_path
        for node in nodes_stack:
            if node.on_blas_path:
                final_nodes_stack.append(node)

    else:
        final_nodes_stack = nodes_stack

    for node in final_nodes_stack:
        print_node(oup, node)

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
    import sys
    from argparse import *

    aparser = ArgumentParser()
    aparser.add_argument('ltrace_file')
    aparser.add_argument('-t', '--trim', action='store_true', help='Only output nodes on a BLAS path')

    args = aparser.parse_args()
    parse_input(args.ltrace_file, args.trim)