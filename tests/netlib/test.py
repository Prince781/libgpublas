#!/usr/bin/env python3

import subprocess
import sys
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=sys.argv[0])
    parser.add_argument('test', help='name of test')
    parser.add_argument('binary', help='location of test binary')
    parser.add_argument('input', help='input file to test binary')
    parser.add_argument('gpublas', help='location of libgpublas')
    parser.add_argument('--nointeractive', action='store_true')

    args = parser.parse_args()
    args.input = os.path.abspath(args.input)
    args.binary = os.path.abspath(args.binary)
    args.gpublas = os.path.abspath(args.gpublas)

    subprocess.run(['rm', '-f', f'{args.test.upper()}.SUMM'], check=True)
    if args.nointeractive:
        subprocess.run(['sh', '-c', f'env LD_PRELOAD={args.gpublas} {args.binary} <{args.input}'], check=True)
    else:
        subprocess.run(['gdb', args.binary, '-ex', f'set exec-wrapper env LD_PRELOAD={args.gpublas}', '-ex', f'run <{args.input}'], check=True)
        subprocess.run(['edit', f'{args.test.upper()}.SUMM'], check=True)
    print(f'Done testing {args.test}')
