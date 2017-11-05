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

echo "malloc" | cat - <(env LD_PRELOAD=$LIBOBJTRACKER $1 | awk '/ptr=\[0x[0-9a-f]+\]/{print "ptr="$2,$3,$4}') | tee $fname

printf "Saved to $fname\n"
