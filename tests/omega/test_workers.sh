#!/bin/bash

python $DGL_HOME/python/dgl/omega/master.py --num_machines 1 --num_gpus_per_machine 8 &

python $DGL_HOME/tests/omega/test_worker.py --num_machines 1 --machine_rank 0 --num_gpus_per_machine 8 --gpu_rank 0 &
python $DGL_HOME/tests/omega/test_worker.py --num_machines 1 --machine_rank 0 --num_gpus_per_machine 8 --gpu_rank 1 &
python $DGL_HOME/tests/omega/test_worker.py --num_machines 1 --machine_rank 0 --num_gpus_per_machine 8 --gpu_rank 2 &
python $DGL_HOME/tests/omega/test_worker.py --num_machines 1 --machine_rank 0 --num_gpus_per_machine 8 --gpu_rank 3 &
python $DGL_HOME/tests/omega/test_worker.py --num_machines 1 --machine_rank 0 --num_gpus_per_machine 8 --gpu_rank 4 &
python $DGL_HOME/tests/omega/test_worker.py --num_machines 1 --machine_rank 0 --num_gpus_per_machine 8 --gpu_rank 5 &
python $DGL_HOME/tests/omega/test_worker.py --num_machines 1 --machine_rank 0 --num_gpus_per_machine 8 --gpu_rank 6 &
python $DGL_HOME/tests/omega/test_worker.py --num_machines 1 --machine_rank 0 --num_gpus_per_machine 8 --gpu_rank 7
