#!/bin/sh
# blas2cuda.sh
# preload libblas2cuda.so and run with a tracker file

trackfile=$1
progname=$2
BLAS2CUDA=./libblas2cuda.so

if [ -z $trackfile ] || [ -z $progname ]; then
    printf "Usage: $0 <objtrackfile> <command>\n"
    exit 1
fi

if [ ! -e $BLAS2CUDA ]; then
    make $BLAS2CUDA
fi

env LD_PRELOAD=$BLAS2CUDA BLAS2CUDA_OPTIONS="track=$trackfile" $progname
