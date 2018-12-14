#!/usr/bin/env python3

import sys
import gzip
import re

if len(sys.argv) < 2:
    sys.exit (f"Usage: {sys.argv[0]} <file>")

records = {}
earliest = None
events = {}
numitems_host = 0
totalsize_host = 0
numitems_dev = 0
totalsize_dev = 0
lineno = 0
failed_to_match = []

class Event:
    def __init__(self, numhost, numdev, sizehost, sizedev):
        self.numhost = numhost
        self.numdev = numdev
        self.sizehost = sizehost
        self.sizedev = sizedev

    def __str__(self):
        return f'Host[{self.numhost} items and {self.sizehost} B] Dev[{self.numdev} items and {self.sizedev} B]'

class Record:
    def __init__(self, nth, fun, reqsize, start, uid, is_gpu=False):
        self.nth = nth
        self.fun = fun
        self.reqsize = reqsize
        self.alive = (start, None)
        self.uid = uid
        self.is_gpu=is_gpu

    def __str__(self):
        return f'{"D" if self.is_gpu else "H"} #{self.nth} U{self.uid} fun=[{self.fun}] reqsize=[{self.reqsize}] alive={self.alive}'

for line in gzip.open(sys.argv[1], 'rt'):
    lineno = lineno + 1
    m = re.match(r'([TUC]) #(\d+).*fun=\[(\w+)\] reqsize=\[(\d+)\].*time=\[(\d+)s\+(\d+)ns\] uid=\[(\d+)\]', line)
    if m:
        tp, nth, fun, reqsize, time_s, time_ns, uid = m.group(1,2,3,4,5,6,7)
        time = int(int(time_s) * 10e9 + int(time_ns))
        if tp == 'T':
            numitems_host += 1
            totalsize_host += int(reqsize)
            records[int(uid)] = Record(nth, fun, int(reqsize), time, int(uid))
            events[time] = Event(numitems_host, numitems_dev, totalsize_host, totalsize_dev)
            if not earliest or time < earliest:
                earliest = time
        elif tp == 'U':
            if not int(uid) in records:
                continue
            if records[int(uid)].is_gpu:
                numitems_dev -= 1
                totalsize_dev -= int(reqsize)
            else:
                numitems_host -= 1
                totalsize_host -= int(reqsize)
            records[int(uid)].alive = (records[int(uid)].alive[0], time)
            events[time] = Event(numitems_host, numitems_dev, totalsize_host, totalsize_dev)
        elif tp == 'C':
            if not int(uid) in records:
                continue
            if not records[int(uid)].is_gpu:
                records[int(uid)].is_gpu = True
                numitems_host -= 1
                totalsize_host -= int(reqsize)
                numitems_dev += 1
                totalsize_dev += int(reqsize)
                for key in events:
                    if key >= records[int(uid)].alive[0]:
                        events[key].numhost -= 1
                        events[key].numdev += 1
                        events[key].sizehost -= records[int(uid)].reqsize
                        events[key].sizedev += records[int(uid)].reqsize
    else:
        failed_to_match.append(lineno)

with gzip.open('failed-to-match.txt.gz', 'wt') as f:
    f.write(f'Failed to match {len(failed_to_match)} items from {sys.argv[1]}. Below are their line numbers:\n')
    for lno in failed_to_match:
        f.write(f'{lno}\n')

for key in records:
    start, end = records[key].alive
    records[key].alive = (start - earliest, (end - earliest) if end else None)

events = {int(key) - int(earliest) : val for key,val in events.items()}

with gzip.open('trace.txt.gz', 'wt') as f:
    for key, value in records.items():
        f.write(f'{value}\n')

with gzip.open('trace.csv.gz', 'wt') as csv:
    csv.write('Created (ns), Destroyed (ns), Host, Size (B)\n')
    for key, value in records.items():
        csv.write(f'{value.alive[0]}, {value.alive[1]}, {not value.is_gpu}, {value.reqsize}\n')

with gzip.open('events.csv.gz', 'wt') as csv:
    with gzip.open('events.txt.gz', 'wt') as f:
        csv.write('Time (ns), Host Items, Host Size (B), Dev Items, Dev Size (B)\n')
        for key, value in events.items():
            f.write(f't={int(key)}ns: {value}\n')
            csv.write(f'{int(key)}, {value.numhost}, {value.sizehost}, {value.numdev}, {value.sizedev}\n')
