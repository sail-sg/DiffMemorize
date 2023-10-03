# This script is to reproduce our Table 1 and Table 2


# Step I: Choose the data size and then generated a class-balanced dataset
size=$1
python data_utils/dataset_size.py --source=datasets/cifar10/cifar-10-python.tar.gz --dest=datasets/cifar10/data_size/cifar10-$size.zip --max-images=$size
python fid.py ref --data=datasets/cifar10/data_size/cifar10-$size.zip --dest=fid-refs/data_size/cifar10-$size.npz


# Step II: Train a diffusion model following basic setup
savedir=$2
num_blocks=2
num_channels=128
batch=$3
lr=$4
weight_decay=$5
torchrun --nproc_per_node 1 \
         --nnodes $WORLD_SIZE \
         --node_rank $RANK \
         --master_addr $MASTER_ADDR \
         --master_port $MASTER_PORT \
         train.py --outdir=$savedir --data=datasets/cifar10/data_size/cifar10-$size.zip --cond=0 --arch=ddpmpp --augment=0.0 --window-size=0.0 --precond=vp --seed=1024 --duration=2000 --num-blocks=$num_blocks --num-channels=$num_channels --fp16=False --lr=$lr --batch=$batch --weight-decay=$weight_decay

# Step III: Generate 10k samples for each model snapshot and evaluate the memorization
outdir=$savedir/00000-cifar10-$size-uncond-ddpmpp-vp-gpus8-batch$batch-fp32
torchrun --nproc_per_node 1 \
         --nnodes $WORLD_SIZE \
         --node_rank $RANK \
         --master_addr $MASTER_ADDR \
         --master_port $MASTER_PORT \
         generate_and_eval_ratio.py --expdir=$outdir --fid-ref=fid-refs/data_size/cifar10-$size.npz --knn-ref=datasets/cifar10/data_size/cifar10-$size.zip --log=$outdir/metrics_top2.log --seeds=0-9999 --subdirs --batch=512