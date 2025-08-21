export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
torchrun --standalone --nproc_per_node=4 train.py
