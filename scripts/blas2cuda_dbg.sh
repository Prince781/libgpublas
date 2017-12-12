#!/bin/bash
# Run the program through gdb

pardir=$(dirname $(readlink -f $0))
debugcmdfile=$1
trackfile=$2
progname=$3
BLAS2CUDA=$pardir/../libblas2cuda.so
optirun=
tempfile=

if [ -z $debugcmdfile ] || [ -z $trackfile ] || [ -z $progname ]; then
    echo "Usage: $0 <debugcmdfile> <objtrackfile> <command> [ARGS...]"
    exit 1
fi

if [ ! -e $BLAS2CUDA ]; then
    if ! make -C $(dirname $BLAS2CUDA); then
        exit 1
    fi
fi

if [ -e $optirun ]; then
    optirun=/bin/optirun
fi

tempfile=$(mktemp)
echo "creating $tempfile"
cat >$tempfile <<EOF
set exec-wrapper $optirun env 'LD_PRELOAD=$pardir/../libblas2cuda.so' 'BLAS2CUDA_OPTIONS=track=$trackfile'
run ${@:4}
$(cat $debugcmdfile)
run ${@:4}
EOF

gdb $progname -x $tempfile
rm $tempfile
