# Unconditional v.s. conditional generation

## Conditional generation with true labels
We first sample a series of training datasets $\mathcal{D}$ using following commands:

```
python data_utils/dataset_size.py --source=datasets/cifar10/cifar-10-python.tar.gz --dest=datasets/cifar10/data_size/cifar10-$size.zip --max-images=$size
```
Here we use the argument `$size` to control the size, i.e., $|\mathcal{D}|$ of the generated class-balanced dataset.

Run the following command to train a conditional diffusion model on $\mathcal{D}$. After training, the saved model snapshots will be saved to `outdir=$savedir/00000-cifar10-$size-cond-ddpmpp-vp-gpus8-batch512-fp32`.
```
torchrun --standalone --nproc_per_node=8 train.py --outdir=$savedir --data=datasets/cifar10/data_size/cifar10-$size.zip --cond=1 --arch=ddpmpp --augment=0.0 --window-size=0.0 --precond=vp --seed=1024 --duration=2000 --num-blocks=2 --num-channels=128 --fp16=False --lr=2e-4
```

## Conditional generation with random labels
We sample a series of training datasets $\mathcal{D}$ with random labeling using following commands:
```
python data_utils/dataset_random.py --source=datasets/cifar10/cifar-10-python.tar.gz --dest=datasets/cifar10/random_cond/cifar10-$num_classes-$size.zip --max-images=$size --num-classes=$num_classes
```
Here we use the argument `$num_classes` to control the number of classes for random labels, i.e. $C$.

Run the following command to train a conditional diffusion model on $\mathcal{D}$ with different $C$. After training, the saved model snapshots will be saved to `outdir=$savedir/00000-cifar10-$num_classes-$size-cond-ddpmpp-vp-gpus8-batch512-fp32`.
```
torchrun --standalone --nproc_per_node=8 train.py --outdir=$savedir --data=datasets/cifar10/random_cond/cifar10-$num_classes-$size.zip --cond=1 --arch=ddpmpp --augment=0.0 --window-size=0.0 --precond=vp --seed=1024 --duration=2000 --num-blocks=2 --num-channels=128 --fp16=False --lr=2e-4
```

## Conditional generation with unique labels
We sample a series of training datasets $\mathcal{D}$ with unique labeling using following commands:
```
python data_utils/dataset_unique.py --source=datasets/cifar10/cifar-10-python.tar.gz --dest=datasets/cifar10/random_cond/cifar10-unique-$size.zip --max-images=$size
```

Then run the following command to train a conditional diffusion model on $\mathcal{D}$ with $C=|\mathcal{D}|$.
```
torchrun --standalone --nproc_per_node=8 train.py --outdir=$savedir --data=datasets/cifar10/random_cond/cifar10-unique-$size.zip --cond=1 --arch=ddpmpp --augment=0.0 --window-size=0.0 --precond=vp --seed=1024 --duration=2000 --num-blocks=2 --num-channels=128 --fp16=False --lr=2e-4
```

Finally, we run the experiments to compare unconditional EDM and conditional EDM with unique labels under $|\mathcal{D}|=50\text{k}$.

* Unconditional EDM:
```
torchrun --standalone --nproc_per_node=8 train.py --outdir=$savedir --data=datasets/cifar10/data_size/cifar10-50000.zip --cond=0 --arch=ddpmpp --augment=0.0 --window-size=0.0 --precond=edm --seed=1024 --duration=600 --num-blocks=4 --num-channels=128 --fp16=False --lr=10e-4
```
After training, the saved model snapshots will be saved to `outdir=$savedir/00000-cifar10-50000-cond-ddpmpp-edm-gpus8-batch512-fp32`.

* Conditional EDM with unique labels
```
torchrun --standalone --nproc_per_node=8 train.py --outdir=$savedir --data=datasets/cifar10/random_cond/cifar10-unique-50000.zip --cond=1 --arch=ddpmpp --augment=0.0 --window-size=0.0 --precond=edm --seed=1024 --duration=600 --num-blocks=4 --num-channels=128 --fp16=False --lr=10e-4
```
After training, the saved model snapshots will be saved to `outdir=$savedir/00000-cifar10-unique-50000-cond-ddpmpp-edm-gpus8-batch512-fp32`.