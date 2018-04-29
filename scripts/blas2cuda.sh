#!/bin/sh
# blas2cuda.sh
# preload libblas2cuda.so and run with a tracker file

pardir=$(dirname $(readlink -f $0))
heuristic=$1
progname=$2
BLAS2CUDA=$pardir/../build/libblas2cuda.so
optirun=

if [ -z $heuristic ] || [ -z $progname ]; then
    echo "Usage: $0 <heuristic> <command> ARGS..."
    exit 1
fi

if [ ! -e $BLAS2CUDA ]; then
    meson $(dirname $BLAS2CUDA)
    if ! ninja -C $(dirname $BLAS2CUDA); then
        exit 1
    fi
fi

if [ -e /bin/optirun ]; then
    optirun=/bin/optirun
fi

$optirun env LD_PRELOAD=$BLAS2CUDA BLAS2CUDA_OPTIONS="heuristic=$heuristic" $progname ${@:3}
