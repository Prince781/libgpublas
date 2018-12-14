#!/bin/sh
# gpublas.sh
# preload libgpublas.so and run with a tracker file

pardir=$(dirname $(readlink -f $0))
heuristic=$1
progname=$2
gpublas=$pardir/../build/libgpublas.so
optirun=

if [ -z $heuristic ] || [ -z $progname ]; then
    echo "Usage: $0 <heuristic> <command> ARGS..."
    exit 1
fi

if [ ! -e $gpublas ]; then
    meson $(dirname $gpublas)
    if ! ninja -C $(dirname $gpublas); then
        exit 1
    fi
fi

if [ -e /bin/optirun ]; then
    optirun=/bin/optirun
fi

$optirun env LD_PRELOAD=$gpublas GPUBLAS_OPTIONS="heuristic=$heuristic" $progname ${@:3}
