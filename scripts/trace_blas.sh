#!/bin/sh

if [ -z $1 ]; then
    echo "Usage: $0 <program> [ARGS...]"
    exit 1
fi

logfile=`basename $1`.ltrace

ltrace -i -e "cblas_*" -e free -e malloc -o "$logfile" $1 ${@:2}
