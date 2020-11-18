#!/bin/bash
NUM_PROC=$1
shift
echo "Executing num proc $NUM_PROC"
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC train.py "$@"

