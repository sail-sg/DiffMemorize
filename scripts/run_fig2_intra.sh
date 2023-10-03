# This script is to reproduce our Fig.2(c)


# Step I: Choose the data size with different intra-class diversity
# prepare ImageNet dataset
python data_utils/sample_imagenet.py
python data_utils/intra_diversity.py --source=datasets/imagenet/dog_2000.txt --dest=datasets/imagenet/imagenet_dog_2000-32x32.zip --resolution=32x32 --transform=center-crop

size=$1
alpha=$2
python data_utils/intra_diversity.py --source=datasets/cifar10/cifar10_dog_2000-32x32.zip --dest=datasets/cifar10/intra_diversity/mixture_$alpha'_'$size.zip --max-images=$size --inter-rate=$alpha
python fid.py ref --data=datasets/cifar10/intra_diversity/mixture_$alpha'_'$size.zip --dest=fid-refs/intra_diversity/mixture_$alpha'_'$size.npz

# Step II: Train a diffusion model following basic setup
savedir=$3
num_blocks=2
num_channels=128
torchrun --nproc_per_node 1 \
         --nnodes $WORLD_SIZE \
         --node_rank $RANK \
         --master_addr $MASTER_ADDR \
         --master_port $MASTER_PORT \
         train.py --outdir=$savedir --data=datasets/cifar10/intra_diversity/mixture_$alpha'_'$size.zip --cond=0 --arch=ddpmpp --augment=0.0 --window-size=0.0 --precond=vp --seed=1024 --duration=2000 --num-blocks=$num_blocks --num-channels=$num_channels --fp16=False --lr=2e-4

# Step III: Generate 10k samples for each model snapshot and evaluate the memorization
outdir=$savedir/00000-mixture_$alpha_$size-uncond-ddpmpp-vp-gpus8-batch512-fp32
torchrun --nproc_per_node 1 \
         --nnodes $WORLD_SIZE \
         --node_rank $RANK \
         --master_addr $MASTER_ADDR \
         --master_port $MASTER_PORT \
         generate_and_eval_ratio.py --expdir=$outdir --fid-ref=fid-refs/intra_diversity/mixture_$alpha'_'$size.npz --knn-ref=datasets/cifar10/intra_diversity/mixture_$alpha'_'$size.zip --log=$outdir/metrics_top2.log --seeds=0-9999 --subdirs --batch=512