#!/bin/sh
# gpublas.sh
# preload libgpublas.so and run with a tracker file

pardir=$(dirname $(readlink -f $0))
trackfile=$1
progname=$2
gpublas=$pardir/../libgpublas.so
optirun=

if [ -z $trackfile ] || [ -z $progname ]; then
    echo "Usage: $0 <objtrackfile> <command> ARGS..."
    exit 1
fi

if [ ! -e $gpublas ]; then
    if ! make -C $(dirname $gpublas); then
        exit 1
    fi
fi

if [ -e /bin/optirun ]; then
    optirun=/bin/optirun
fi

$optirun env LD_PRELOAD=$gpublas GPUBLAS_OPTIONS="track=$trackfile" nvprof $progname ${@:3}
