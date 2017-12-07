#!/bin/bash
# Run the program through gdb

debugcmdfile=$1
trackfile=$2
progname=$3
BLAS2CUDA=../libblas2cuda.so
tempfile=

if [ -z $debugcmdfile ] || [ -z $trackfile ] || [ -z $progname ]; then
    echo "Usage: $0 <debugcmdfile> <objtrackfile> <command> [ARGS...]"
    exit 1
fi

if [ ! -e $BLAS2CUDA ]; then
    make -C .. $BLAS2CUDA
fi

tempfile=$(mktemp)
echo "creating $tempfile"
cat >$tempfile <<EOF
set environment LD_PRELOAD=../libblas2cuda.so
set environment BLAS2CUDA_OPTIONS=track=$trackfile
run ${@:4}
$(cat $debugcmdfile)
run ${@:4}
EOF

gdb $progname -x $tempfile
rm $tempfile
