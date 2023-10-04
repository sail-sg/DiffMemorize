# Training procedure

Before conducting the following experiments, we first sample a series of training datasets $\mathcal{D}$ using following commands:

```
python data_utils/dataset_size.py --source=datasets/cifar10/cifar-10-python.tar.gz --dest=datasets/cifar10/data_size/cifar10-$size.zip --max-images=$size
```
Here we use the argument `$size` to control the size, i.e., $|\mathcal{D}|$ of the generated class-balanced dataset.

## Batch size
Run the following command to train a diffusion model on $\mathcal{D}$ with different batch sizes and learning rates. After training, the saved model snapshots will be saved to `outdir=$savedir/00000-cifar10-$size-uncond-ddpmpp-vp-gpus8-batch$batch-fp32`.
```
torchrun --standalone --nproc_per_node=8 train.py --outdir=$savedir --data=datasets/cifar10/data_size/cifar10-$size.zip --cond=0 --arch=ddpmpp --augment=0.0 --window-size=0.0 --precond=vp --seed=1024 --duration=2000 --num-blocks=2 --num-channels=128 --fp16=False --lr=$lr --batch=$batch
```

Here, to adjust the batch size `$batch`, we also need to adjust the learning rate accordingly to ensure a consistent ratio of learning rate to batch size, which is $2\times10^{-4}/512$. Specifically, the pair of `$batch` and `$lr` is ranged from $\{(128, 0.5\times10^{-4}), (256, 1.0\times10^{-4}), (384, 1.5\times10^{-4}), (512, 2.0\times10^{-4}), (640, 2.5\times10^{-4}), (768, 3.0\times10^{-4}), (896, 3.5\times10^{-4})\}$.

## Weight decay
Run the following command to train a diffusion model on $\mathcal{D}$ with different weight decays. After training, the saved model snapshots will be saved to `outdir=$savedir/00000-cifar10-$size-uncond-ddpmpp-vp-gpus8-batch512-fp32`.
```
torchrun --standalone --nproc_per_node=8 train.py --outdir=$savedir --data=datasets/cifar10/data_size/cifar10-$size.zip --cond=0 --arch=ddpmpp --augment=0.0 --window-size=0.0 --precond=vp --seed=1024 --duration=2000 --num-blocks=2 --num-channels=128 --fp16=False --lr=2e-4 --weight-decay=$weight_decay
```
Here, `$weight_decay` is ranged from $\{0.0, 1\times10^{-5}, 1\times10^{-4}, 1\times10^{-3}, 1\times10^{-2}, 2\times10^{-2}, 5\times10^{-2}, 8\times10^{-2}, 1\times10^{-1}\}$.

## EMA
Run the following command to train a diffusion model on $\mathcal{D}$ with different EMA ratios. After training, the saved model snapshots will be saved to `outdir=$savedir/00000-cifar10-$size-uncond-ddpmpp-vp-gpus8-batch512-fp32`.
```
torchrun --standalone --nproc_per_node=8 train.py --outdir=$savedir --data=datasets/cifar10/data_size/cifar10-$size.zip --cond=0 --arch=ddpmpp --augment=0.0 --window-size=0.0 --precond=vp --seed=1024 --duration=2000 --num-blocks=2 --num-channels=128 --fp16=False --lr=2e-4 --final-ema=$ema --ema_mode=mode4
```

Here, `$ema` is ranged from $0.99929, 0.999, 0.99, 0.9, 0.8, 0.5, 0.2, 0.1, 0.0$.