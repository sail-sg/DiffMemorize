# This script is to reproduce our Fig1. (c) and Fig1. (d)


# Step I: Choose the data size and then generated a class-balanced dataset
size=$1
python data_utils/dataset_size.py --source=datasets/cifar10/cifar-10-python.tar.gz --dest=datasets/cifar10/data_size/cifar10-$size.zip --max-images=$size
python fid.py ref --data=datasets/cifar10/data_size/cifar10-$size.zip --dest=fid-refs/data_size/cifar10-$size.npz


# Step II: Train a diffusion model following EDM
savedir=$2
duration=$3
torchrun --nproc_per_node 1 \
         --nnodes $WORLD_SIZE \
         --node_rank $RANK \
         --master_addr $MASTER_ADDR \
         --master_port $MASTER_PORT \
         train.py --outdir=$savedir --data=datasets/cifar10/data_size/cifar10-$size.zip --cond=0 --arch=ddpmpp --augment=0.0 --window-size=0.0 --precond=edm --seed=1024 --duration=$duration --num-blocks=4 --num-channels=128 --fp16=False --lr=10e-4

# Step III: Generate 10k samples for each model snapshot and evaluate the memorization
outdir=$savedir/00000-cifar10-$size-uncond-ddpmpp-edm-gpus8-batch512-fp32
torchrun --nproc_per_node 1 \
         --nnodes $WORLD_SIZE \
         --node_rank $RANK \
         --master_addr $MASTER_ADDR \
         --master_port $MASTER_PORT \
         generate_and_eval_ratio.py --expdir=$outdir --fid-ref=fid-refs/data_size/cifar10-$size.npz --knn-ref=datasets/cifar10/data_size/cifar10-$size.zip --log=$outdir/metrics_top2.log --seeds=0-9999 --subdirs --batch=512