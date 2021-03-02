#!/bin/bash
NUM_PROC=$1
shift
echo "Executing num proc $NUM_PROC"
echo "$@"

SCRIPT=$(readlink -f "$0")
SCRIPT_PATH=$(dirname "$SCRIPT")

python -m torch.distributed.launch --nproc_per_node=$NUM_PROC $SCRIPT_PATH/train.py "$@"

