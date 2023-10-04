# Data Distribution

## Data dimension
We first sample a series of training datasets $\mathcal{D}$ with different dimensions using following commands:
```
python data_utils/dataset_tool.py --source=datasets/cifar10/cifar-10-python.tar.gz --dest=datasets/cifar10/data_dimension/cifar10-$resolution'x'$resolution'-'$size.zip --resolution=$resolution'x'$resolution
```
Here we use the argument `$size` to control the size, i.e., $|\mathcal{D}|$ of the generated class-balanced dataset. The argument `$resolution` refers to the spatial dimension, which can be $32$ or $16$ or $8$.

Run the following command to train a diffusion model on $\mathcal{D}$ with different spatial resolutions. After training, the saved model snapshots will be saved to `outdir=$savedir/00000-cifar10-$resolution'x'$resolution'-'$size-uncond-ddpmpp-vp-gpus8-batch512-fp32`.
```
torchrun --standalone --nproc_per_node=8 train.py --outdir=$savedir --data=datasets/cifar10/data_dimension/cifar10-$resolution'x'$resolution'-'$size.zip --cond=0 --arch=ddpmpp --augment=0.0 --window-size=0.0 --precond=vp --seed=1024 --duration=2000 --num-blocks=2 --num-channels=128 --fp16=False --lr=2e-4 --attn-resolutions=$attn_resolutions
```

Here `$attn_resolutions` refers to the location of attention mechanism in the model architecture. We take the following values to ensure the model architecture the same.

* When `$resolution=32`, `$attn_resolutions=16`.
* When `$resolution=16`, `$attn_resolutions=8`.
* When `$resolution=8`, `$attn_resolutions=4`. 

## Number of classes
We sample a series of training datasets $\mathcal{D}$ with different inter-diversity using following commands:
```
python data_utils/dataset_diversity.py --source=datasets/cifar10/cifar-10-python.tar.gz --dest=datasets/cifar10/data_diversity/cifar10-$num_classes-$size.zip --max-images=$size --num-classes=$num_classes
```
Here we use the argument `$num_classes` to control the number of classes, i.e. $C$. Here `$num_classes` ranges from the set $\{1, 2, 5, 10\}$.

Then run the following command to train a diffusion model on $\mathcal{D}$ with different $C$. After training, the saved model snapshots will be saved to `outdir=$savedir/00000-cifar10-$num_classes-$size-uncond-ddpmpp-vp-gpus8-batch512-fp32`.
```
torchrun --standalone --nproc_per_node=8 train.py --outdir=$savedir --data=datasets/cifar10/data_diversity/cifar10-$num_classes-$size.zip --cond=0 --arch=ddpmpp --augment=0.0 --window-size=0.0 --precond=vp --seed=1024 --duration=2000 --num-blocks=2 --num-channels=128 --fp16=False --lr=2e-4
```

## Intra-diversity
To control the intra-diversity of data distribution, we need to mix the data from both CIFAR-10 and ImageNet datasets, where are saved in `datasets/cifar10` and `datasets/imagenet`.

We first sample 2k images of dog class in ImageNet by running the following commands. The images will be resized to a spatial resolution of $32\times32$.
```
python data_utils/sample_imagenet.py
python data_utils/intra_diversity.py --source=datasets/imagenet/dog_2000.txt --dest=datasets/imagenet/imagenet_dog_2000-32x32.zip --resolution=32x32 --transform=center-crop
```

Afterwards, we blend images from ImageNet to CIFAR-10.
```
python data_utils/intra_diversity.py --source=datasets/cifar10/cifar10_dog_2000-32x32.zip --dest=datasets/cifar10/intra_diversity/mixture_$alpha'_'$size.zip --max-images=$size --inter-rate=$alpha
```

Here `$alpha` represents the proportion of ImageNet data in the constructed training dataset, ranged from $\{0.0, 0.2, 0.5, 0.8, 1.0\}$.

Run the following command to train a diffusion model on $\mathcal{D}$ with different $C$. After training, the saved model snapshots will be saved to `outdir=$savedir/00000-mixture_$alpha_$size-uncond-ddpmpp-vp-gpus8-batch512-fp32`.
```
torchrun --standalone --nproc_per_node=8 train.py --outdir=$savedir --data=datasets/cifar10/intra_diversity/mixture_$alpha'_'$size.zip --cond=0 --arch=ddpmpp --augment=0.0 --window-size=0.0 --precond=vp --seed=1024 --duration=2000 --num-blocks=2 --num-channels=128 --fp16=False --lr=2e-4
```