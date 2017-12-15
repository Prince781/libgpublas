#!/bin/sh
# blas2cuda.sh
# preload libblas2cuda.so and run with a tracker file

pardir=$(dirname $(readlink -f $0))
trackfile=$1
progname=$2
BLAS2CUDA=$pardir/../libblas2cuda.so
optirun=

if [ -z $trackfile ] || [ -z $progname ]; then
    echo "Usage: $0 <objtrackfile> <command> ARGS..."
    exit 1
fi

if [ ! -e $BLAS2CUDA ]; then
    if ! make -C $(dirname $BLAS2CUDA); then
        exit 1
    fi
fi

if [ -e /bin/optirun ]; then
    optirun=/bin/optirun
fi

$optirun env LD_PRELOAD=$BLAS2CUDA BLAS2CUDA_OPTIONS="track=$trackfile" nvprof $progname ${@:3}
