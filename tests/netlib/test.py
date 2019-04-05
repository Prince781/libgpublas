#!/usr/bin/env python3

import subprocess
import sys

if __name__ == '__main__':
    test_name = (len(sys.argv) == 2 and sys.argv[1]) or 'dblat3'
    subprocess.run(['rm', '-f', f'{test_name.upper()}.SUMM'], check=True)
    subprocess.run(['gdb', f'build/{test_name}-blas', '-ex', 'set exec-wrapper env LD_PRELOAD=../../build/libblas2cuda.so', '-ex', f'run <input.{test_name}'], check=True)
    subprocess.run(['edit', f'{test_name.upper()}.SUMM'], check=True)
    print(f'Done testing {test_name}')
