#!/bin/bash
# :vim tw=0
# Run the object tracker through gdb

pardir=$(dirname $(readlink -f $0))
debugcmdfile=$1
blas_lib=$2
program=$3
args=${@:4}
fname=$(basename $program.objtrack)
LIBOBJTRACKER=$pardir/../lib/libobjtracker.so


if [ -z $debugcmdfile ] || [ -z $blas_lib ] || [ -z $program ]; then
    printf "Usage: $0 <debugcmdfile> <blas_lib1.so[,blas_lib2.so,...]> <program> ARGS...\n"
    exit 1
fi

if [ ! -e $LIBOBJTRACKER ]; then
    make -C $(dirname $LIBOBJTRACKER) $(basename $LIBOBJTRACKER)
fi

tempfile=$(mktemp)
echo "creating $tempfile"

cat >$tempfile <<EOF
set exec-wrapper env 'OBJTRACKER_OPTIONS=blas_libs=$blas_lib' 'LD_PRELOAD=$LIBOBJTRACKER'
run $args
$(cat $debugcmdfile)
run $args
EOF

gdb $program -x $tempfile
rm $tempfile
