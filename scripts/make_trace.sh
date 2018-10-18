#!/bin/sh

exe=$1
libs=(${@:2})

if [ -z "$exe" ]; then
    echo "Usage: $0 exe [library]..."
    exit 1
fi

funcs=()

tempfile=$(mktemp gdb-cmds.XXXXXXXX --tmpdir)
tempfile2=$(mktemp)
echo "creating $tempfile and $tempfile2 ..."

if [ ! -e $tempfile ] || [ ! -e $tempfile2 ]; then
    echo "Failed"
    exit 1
fi

cat >$tempfile <<EOF
set logging on
set confirm off
EOF

cat >$tempfile2 <<EOF
command
silent
backtrace 1
continue
end
EOF

nm "$exe" | grep -E ' [TW] ' | awk '{print "break", $3}' | cat - $tempfile2 >> $tempfile

for lib in ${libs[@]}; do
    nm "$lib" | grep -E ' [TW] ' | awk '{print "break", $3}' | \
    while read line; do
        echo $line | cat - $tempfile2 >> $tempfile
    done
done

read -p "Delete temp file? ($tempfile) [y/[n]]: " delete_temp_file
if [[ $delete_temp_file == y ]]; then
    rm -f $tempfile
fi
rm -f $tempfile2
