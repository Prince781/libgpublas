#!/bin/sh

if [ -z $1 ]; then
    echo "Usage: $0 <program> [ARGS...]"
    exit 1
fi

ltrace -e "cblas_*" -e free -e malloc $1 ${@:2}
