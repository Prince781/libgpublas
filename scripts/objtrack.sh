#!/bin/bash
# :vim tw=0
# objtrack.sh
# Track invocations of a program and produce a format
# readable by the object tracker

pardir=$(dirname $(readlink -f $0))

if [ -z $1 ] || [ -z $2 ]; then
    printf "Usage: $0 <blas_lib1.so[,blas_lib2.so,...]> <program> ARGS...\n"
    exit 1
fi

fname=$(basename ${2}.objtrack)
LIBOBJTRACKER=$pardir/../lib/libobjtracker.so

if [ ! -e $LIBOBJTRACKER ]; then
    make -C $(dirname $LIBOBJTRACKER) $(basename $LIBOBJTRACKER)
fi

echo "running: env OBJTRACKER_OPTIONS=\"blas_libs=$1\" LD_PRELOAD=$LIBOBJTRACKER $2 ${@:3}"

cat < <(env OBJTRACKER_OPTIONS="blas_libs=$1" LD_PRELOAD=$LIBOBJTRACKER $2 ${@:3} | awk '/C \[0x[0-9a-f]+\].*fun=\[\w+\] reqsize=\[[0-9a-f]+\] ip_offs=\[[0-9a-f\.]+\]/{print $3,$4,$5}' | sort -u) | tee $fname

printf "Saved to $fname\n"
