#!/bin/env python3

import sys
import gzip
import re

if len(sys.argv) < 2:
    sys.exit (f"Usage: {sys.argv[0]} <file>")

records = {}
earliest = None
events = {}
numitems = 0
totalsize = 0

class Event(object):
    def __init__(self, numitems, totalsize):
        self.numitems = numitems
        self.totalsize = totalsize

    def __str__(self):
        return f'{self.numitems} items at {self.totalsize} B'

class Record(object):
    def __init__(self, fun, reqsize, start, uid):
        self.fun = fun
        self.reqsize = reqsize
        self.alive = (start, None)
        self.uid = uid

    def __str__(self):
        return f'fun=[{self.fun}] reqsize=[{self.reqsize}] alive={self.alive} uid=[{self.uid}]'

for line in gzip.open(sys.argv[1], 'rt'):
    m = re.match(r'([TU]) fun=\[(\w+)\] reqsize=\[(\d+)\] ip_offs=\[(.*)\] time=\[(\d+)s\+(\d+)ns\] uid=\[(\d+)\]', line)
    if m:
        tp, fun, reqsize, time_s, time_ns, uid = m.group(1,2,3,5,6,7)
        time = int(time_s) * 10e9 + int(time_ns)
        if tp == 'T':
            numitems += 1
            totalsize += int(reqsize)
            records[int(uid)] = Record(fun, int(reqsize), time, int(uid))
            events[time] = Event(numitems, totalsize)
            if not earliest or time < earliest:
                earliest = time
        elif tp == 'U':
            numitems -= 1
            totalsize -= int(reqsize)
            records[int(uid)].alive = (records[int(uid)].alive[0], time)
            events[time] = Event(numitems, totalsize)

for key in records:
    start, end = records[key].alive
    records[key].alive = (start - earliest, (end - earliest) if end else None)

events = dict([(int(key) - int(earliest), val) for key,val in events.items()])

with open('trace.txt', 'w') as f:
    for key, value in records.items():
        f.write(f'{value}\n')

with open('events.txt', 'w') as f:
    for key, value in events.items():
        f.write(f'{value} at t={int(key)}ns\n')
