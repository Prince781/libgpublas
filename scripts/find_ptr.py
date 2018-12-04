#!/usr/bin/env python3

import sys
import re

if len(sys.argv) < 3:
    sys.exit (f"Usage: {sys.argv[0]} ltrace-file address")

lineno = 0

unfinished_stack = []       # sizes
pointer_re = '0x[A-Za-z0-9]+'
addr = int(sys.argv[2], 16)

for line in open(sys.argv[1], 'rt'):
    lineno = lineno + 1

    # malloc()
    m = re.match(fr'.*malloc\((\d+)\).*= ({pointer_re})', line)
    if m:
        sz, ptr = m.group(1,2)
        ptr = int(ptr, 16)
        sz = int(sz)

        if addr >= ptr and addr <= (ptr + sz):
            sys.exit(f'line {lineno}')
        continue

    m = re.match(fr'.*calloc\((\d+), (\d+)\).*= ({pointer_re})', line)
    if m:
        nmemb, sz, ptr = m.group(1,2)
        ptr = int(ptr, 16)
        nmemb = int(nmemb)
        sz = int(sz)

        if addr >= ptr and addr <= (ptr + nmemb*sz):
            sys.exit(f'line {lineno}')
        continue

    # malloc() (unfinished)
    m = re.match(fr'.*malloc\((\d+).*<unfinished ...>', line)
    if m:
        unfinished_stack.append(int(m.group(1)))
        continue

    # calloc() (unfinished)
    m = re.match(fr'.*calloc\((\d+), (\d+).*<unfinished ...>', line)
    if m:
        nmemb, size = m.group(1,2)
        unfinished_stack.append(int(nmemb) * int(size))
        continue

    # realloc() (unfinished)
    m = re.match(fr'.*realloc\(.*, (\d+).*<unfinished ...>', line)
    if m:
        unfinished_stack.append(int(m.group(1)))
        continue

    # *alloc() (resumed)
    m = re.match(fr'.*(?:m|c|re)alloc resumed> \).*= ({pointer_re})', line)
    if m:
        ptr = m.group(1)
        sz = unfinished_stack.pop()
        ptr = int(ptr, 16)

        if addr >= ptr and addr <= (ptr + sz):
            sys.exit(f'line {lineno}')
        continue

sys.exit(f'nothing found')
