#!/bin/bash
# :vim tw=0
# objtrace.sh
# Trace the history of invocations. This includes things like allocations and
# deallocations.

pardir=$(dirname $(readlink -f $0))

if [ -z $1 ] || [ -z $2 ]; then
    printf "Usage: $0 <blas_lib1.so[,blas_lib2.so,...]> <program> ARGS...\n"
    exit 1
fi

fname=$(basename ${2}.objtrace)
LIBOBJTRACKER=$pardir/../build/lib/libobjtracker.so

if [ ! -e $LIBOBJTRACKER ]; then
    meson $(dirname $LIBOBJTRACKER)
    ninja -C $(dirname $LIBOBJTRACKER)
fi

echo "running: env OBJTRACKER_OPTIONS=\"blas_libs=$1\" LD_PRELOAD=$LIBOBJTRACKER $2 ${@:3}"

cat < <(env OBJTRACKER_OPTIONS="blas_libs=$1" LD_PRELOAD=$LIBOBJTRACKER $2 ${@:3} | awk '/[CUT] \[0x[0-9a-f]+\] fun=\[\w+\] reqsize=\[[0-9a-f]+\]/{print $1,$2,$3,$4,$5,$6,$7}' 2>stderr.txt) | tee $fname

gzip -f $fname
printf "Saved to $fname.gz\n"
