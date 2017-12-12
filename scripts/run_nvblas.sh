#!/bin/sh

nvblas_conf=$1
program=$2
args=${@:3}
optirun=

if [ -z $nvblas_conf ] || [ -z $program ]; then
	echo "Usage: $0 <nvblas.conf> <program> ARGS..."
	exit 1
fi

if [ -e /bin/optirun ]; then
	optirun=/bin/optirun
fi

echo "running: $optirun env LD_PRELOAD=$CUDA/lib64/libnvblas.so NVBLAS_CONFIG_FILE=$nvblas_conf $program $args"
$optirun env LD_PRELOAD=$CUDA/lib64/libnvblas.so NVBLAS_CONFIG_FILE=$nvblas_conf $program $args
