#!/bin/bash
# objtrack.sh
# Track invocations of a program and produce a format
# readable by the object tracker

if [ -z $1 ] || [ -z $2 ]; then
    printf "Usage: $0 <blas_lib.so> <program> ARGS...\n"
    exit 1
fi

if [ ! -e $2 ]; then
    echo "$2 doesn't exist"
    exit 1
fi

fname=$(basename ${2}.objtrack)
LIBOBJTRACKER=../lib/libobjtracker.so

if [ ! -e $LIBOBJTRACKER ]; then
    make -C $(dirname $LIBOBJTRACKER) $(basename $LIBOBJTRACKER)
fi

echo "running: env OBJTRACKER_BLASLIB=$1 LD_PRELOAD=$LIBOBJTRACKER $2 ${@:3}"
echo "malloc" | cat - <(env OBJTRACKER_BLASLIB=$1 LD_PRELOAD=$LIBOBJTRACKER $2 ${@:3} | awk '/C \[0x[0-9a-f]+\].*reqsize=\[[0-9a-f]+\]/{print $3,$4}' | sort -u) | tee $fname

printf "Saved to $fname\n"
