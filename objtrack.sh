#!/bin/bash
# objtrack.sh
# Track invocations of a program and produce a format
# readable by the object tracker

if [ -z $1 ]; then
    printf "Usage: $0 <program>\n"
    exit 1
fi

fname=$(basename ${1}.objtrack)
LIBOBJTRACKER=./lib/libobjtracker.so

make -C ./lib/ libobjtracker.so

echo "malloc" | cat - <(env LD_PRELOAD=$LIBOBJTRACKER $1 | awk '/reqsize=\[[0-9a-f]+\]/{print $3,$4}' | sort -u) | tee $fname

printf "Saved to $fname\n"
