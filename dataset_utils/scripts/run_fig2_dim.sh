# This script is to reproduce our Fig.2(a)


# Step I: Choose the data size with different resolution and then generated a class-balanced dataset
size=$1
resolution=$2
python data_utils/dataset_tool.py --source=datasets/cifar10/cifar-10-python.tar.gz --dest=datasets/cifar10/data_dimension/cifar10-$resolution'x'$resolution'-'$size.zip --resolution=$resolution'x'$resolution
python fid.py ref --data=datasets/cifar10/data_dimension/cifar10-$resolution'x'$resolution'-'$size.zip --dest=fid-refs/data_dimension/cifar10-$resolution'x'$resolution'-'$size.npz

# Step II: Train a diffusion model following basic setup
savedir=$3
attn_resolution=$4
num_blocks=2
num_channels=128
torchrun --nproc_per_node 1 \
         --nnodes $WORLD_SIZE \
         --node_rank $RANK \
         --master_addr $MASTER_ADDR \
         --master_port $MASTER_PORT \
         train.py --outdir=$savedir --data=datasets/cifar10/data_dimension/cifar10-$resolution'x'$resolution'-'$size.zip --cond=0 --arch=ddpmpp --augment=0.0 --window-size=0.0 --precond=vp --seed=1024 --duration=2000 --num-blocks=$num_blocks --num-channels=$num_channels --fp16=False --lr=2e-4 --attn-resolutions=$attn_resolutions

# Step III: Generate 10k samples for each model snapshot and evaluate the memorization
outdir=$savedir/00000-cifar10-$resolution'x'$resolution'-'$size-uncond-ddpmpp-vp-gpus8-batch512-fp32
torchrun --nproc_per_node 1 \
         --nnodes $WORLD_SIZE \
         --node_rank $RANK \
         --master_addr $MASTER_ADDR \
         --master_port $MASTER_PORT \
         generate_and_eval_ratio.py --expdir=$outdir --fid-ref=fid-refs/data_dimension/cifar10-$resolution'x'$resolution'-'$size.npz --knn-ref=datasets/cifar10/data_dimension/cifar10-$resolution'x'$resolution'-'$size.zip --log=$outdir/metrics_top2.log --seeds=0-9999 --subdirs --batch=512