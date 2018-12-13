#!/usr/bin/env python3

import sys
import re

if len(sys.argv) < 3:
    sys.exit (f"Usage: {sys.argv[0]} objtrace-file address")

lineno = 0

unfinished_stack = []       # sizes
pointer_re = '0x[A-Za-z0-9]+'
addr = int(sys.argv[2], 16)

for line in open(sys.argv[1], 'rt'):
    lineno = lineno + 1

    # *alloc()
    m = re.match(fr'.*\[({pointer_re})\] fun=\[\walloc\] reqsize=\[(\d+)\].*', line)
    if m:
        ptr, sz = m.group(1,2)
        ptr = int(ptr, 16)
        sz = int(sz)

        if addr >= ptr and addr <= (ptr + sz):
            sys.exit(f'line {lineno}')
        continue

sys.exit(f'nothing found')
