#!/bin/bash
# Run the program through gdb

pardir=$(dirname $(readlink -f $0))
debugcmdfile=$1
heuristic=$2
progname=$3
gpublas=$pardir/../build/libgpublas.so
optirun=
tempfile=

if [ -z $debugcmdfile ] || [ -z $heuristic ] || [ -z $progname ]; then
    echo "Usage: $0 <debugcmdfile> <heuristic> <command> [ARGS...]"
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

tempfile=$(mktemp)
echo "creating $tempfile"
cat >$tempfile <<EOF
set exec-wrapper $optirun env 'LD_PRELOAD=$gpublas' 'GPUBLAS_OPTIONS=heuristic=$heuristic'
run ${@:4}
$(cat $debugcmdfile)
run ${@:4}
EOF

gdb $progname -x $tempfile
rm $tempfile
