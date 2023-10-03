# This script is to reproduce our results for conditional EDM with unique labels


# Step I: Choose the data size and then generated a dataset with unique labels
size=50000
python data_utils/dataset_unique.py --source=datasets/cifar10/cifar-10-python.tar.gz --dest=datasets/cifar10/random_cond/cifar10-unique-$size.zip --max-images=$size
python fid.py ref --data=datasets/cifar10/random_cond/cifar10-unique-$size.zip --dest=fid-refs/random_cond/cifar10-unique-$size.npz


# Step II: Train a diffusion model following basic setup
savedir=$1
num_blocks=4
num_channels=128
torchrun --nproc_per_node 1 \
         --nnodes $WORLD_SIZE \
         --node_rank $RANK \
         --master_addr $MASTER_ADDR \
         --master_port $MASTER_PORT \
         train.py --outdir=$savedir --data=datasets/cifar10/random_cond/cifar10-unique-$size.zip --cond=1 --arch=ddpmpp --augment=0.0 --window-size=0.0 --precond=edm --seed=1024 --duration=600 --num-blocks=$num_blocks --num-channels=$num_channels --fp16=False --lr=10e-4

# Step III: Generate 10k samples for each model snapshot and evaluate the memorization
outdir=$savedir/00000-cifar10-unique-$size-cond-ddpmpp-edm-gpus8-batch512-fp32
torchrun --nproc_per_node 1 \
         --nnodes $WORLD_SIZE \
         --node_rank $RANK \
         --master_addr $MASTER_ADDR \
         --master_port $MASTER_PORT \
         generate_and_eval_ratio.py --expdir=$outdir --fid-ref=fid-refs/random_cond/cifar10-unique-$size.npz --knn-ref=datasets/cifar10/random_cond/cifar10-unique-$size.zip --log=$outdir/metrics_top2.log --seeds=0-9999 --subdirs --batch=512