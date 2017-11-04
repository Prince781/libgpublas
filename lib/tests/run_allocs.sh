#!/bin/sh

rm tree*.dot

gdb ./allocs

rm -rf trees/
mkdir trees/

for f in tree*.dot; do
    /bin/dot -Tpng $f -o trees/$f.png
    mv $f trees/
done > /dev/null    # why is dot printing stuff out to the console?
