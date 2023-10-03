# This script is to generate images by the theoretical optimum and EDM, to reproduce our Fig1. (a) and Fig1. (b)


# Step I: Prepare full CIFAR-1O dataset
python dataset_tool.py --source=datasets/cifar10/cifar-10-python.tar.gz --dest=datasets/cifar10/cifar10-train.zip
python fid.py ref --data=datasets/cifar10/cifar10-train.zip --dest=fid-refs/cifar10-32x32.npz


# Step II: Generate images by the theoretical optimum
torchrun --standalone --nproc_per_node=1 generate_optim.py --outdir=fid-tmp-optim --seeds=0-49999 --subdirs --network=datasets/cifar10/cifar10-train.zip

# Step III: Generate images by EDM
torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp-edm --seeds=0-49999 --subdirs --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl