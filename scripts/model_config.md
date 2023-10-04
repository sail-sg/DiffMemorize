# Model configuration $\mathcal{M}$

Before conducting the following experiments, we first sample a series of training datasets $\mathcal{D}$ using following commands:

```
python data_utils/dataset_size.py --source=datasets/cifar10/cifar-10-python.tar.gz --dest=datasets/cifar10/data_size/cifar10-$size.zip --max-images=$size
```
Here we use the argument `$size` to control the size, i.e., $|\mathcal{D}|$ of the generated class-balanced dataset.


## Model size
Run the following command to train a diffusion model on $\mathcal{D}$ with different model sizes:
```
torchrun --standalone --nproc_per_node=8 train.py --outdir=$savedir --data=datasets/cifar10/data_size/cifar10-$size.zip --cond=0 --arch=ddpmpp --augment=0.0 --window-size=0.0 --precond=vp --seed=1024 --duration=2000 --num-blocks=$num_blocks --num-channels=$num_channels --fp16=False --lr=2e-4
```
In our basic experimental setup, `$num_blocks` is 2 and `$num_channels` is 128. 
* To modify the model width, we select values of `$num_channels` from $\{128, 192, 256, 320\}$ in Section 4.1. 
* To modify the model depths, we select values of `$num_blocks` from $2$ to $12$ in Section 4.1.

After training, the saved model snapshots will be saved to `outdir=$savedir/00000-cifar10-$size-uncond-ddpmpp-vp-gpus8-batch512-fp32`. Then conduct the evaluation using:

```
torchrun --standalone --nproc_per_node=$num_gpu mem_ratio.py --expdir=$outdir --knn-ref=$data_path --log=$outdir/mem_traj.log --seeds=0-9999 --subdirs --batch=512
```


## Time embedding
Run the following command to train a diffusion model on $\mathcal{D}$ with different time embeddings: 
```
torchrun --standalone --nproc_per_node=8 train.py --outdir=$savedir --data=datasets/cifar10/data_size/cifar10-$size.zip --cond=0 --arch=ddpmpp --augment=0.0 --window-size=0.0 --precond=vp --seed=1024 --duration=2000 --num-blocks=$num_blocks --num-channels=128 --embed=$embed --fp16=False --lr=2e-4
```

Here `$embed` can be either `positional` (positional embedding) or `fourier` (fourier random features). In Section 4.2, we modify the value of `$num_blocks` to $2$ or $4$ to conduct our experiments. After training, the saved model snapshots will be saved to `outdir=$savedir/00000-cifar10-$size-uncond-ddpmpp-vp-gpus8-batch512-fp32`.


## Skip connections
Run the following command to train a DDPM++ on $\mathcal{D}$ with different skip connections. After training, the saved model snapshots will be saved to `outdir=$savedir/00000-cifar10-$size-uncond-ddpmpp-vp-gpus8-batch512-fp32`.
```
torchrun --standalone --nproc_per_node=8 train.py --outdir=$savedir --data=datasets/cifar10/data_size/cifar10-$size.zip --cond=0 --arch=ddpmpp --augment=0.0 --window-size=0.0 --precond=vp --seed=1024 --duration=2000 --num-blocks=2 --num-channels=128 --embed=positional --fp16=False --lr=2e-4 --skip-connections=$skip
```

Run the following command to train an NCSN++ on $\mathcal{D}$ with different skip connections. After training, the saved model snapshots will be saved to `outdir=$savedir/00000-cifar10-$size-uncond-ncsnpp-ve-gpus8-batch512-fp32`.
```
torchrun --standalone --nproc_per_node=8 train.py --outdir=$savedir --data=datasets/cifar10/data_size/cifar10-$size.zip --cond=0 --arch=ncsnpp --augment=0.0 --window-size=0.0 --precond=ve --seed=1024 --duration=2000 --num-blocks=2 --num-channels=128 --embed=fourier --fp16=False --lr=2e-4 --skip-connections=$skip
```

`$skip` refers to how to retain specific skip connections.
* Full skip connections: `$skip=1,1,1,1,1,1,1,1,1`
* Retain specific skip connections at resolution of $32\times32$: `$skip=1,1,1,0,0,0,0,0,0` (skip number=3), `$skip=1,1,0,0,0,0,0,0,0`, `$skip=1,0,1,0,0,0,0,0,0`, `$skip=0,1,1,0,0,0,0,0,0` (skip number=2), `$skip=1,0,0,0,0,0,0,0,0`, `$skip=0,1,0,0,0,0,0,0,0`, `$skip=0,0,1,0,0,0,0,0,0` (skip number=1) 
* Retain specific skip connections at resolution of $16\times16$: `$skip=0,0,0,1,1,1,0,0,0` (skip number=3), `$skip=0,0,0,1,1,0,0,0,0`, `$skip=0,0,0,1,0,1,0,0,0`, `$skip=0,0,0,0,1,1,0,0,0` (skip number=2), `$skip=0,0,0,1,0,0,0,0,0`, `$skip=0,0,0,0,1,0,0,0,0`, `$skip=0,0,0,0,0,1,0,0,0` (skip number=1) 
* Retain specific skip connections at resolution of $8\times8$: `$skip=0,0,0,0,0,0,1,1,1` (skip number=3), `$skip=0,0,0,0,0,0,1,1,0`, `$skip=0,0,0,0,0,0,1,0,1`, `$skip=0,0,0,0,0,0,0,1,1` (skip number=2), `$skip=0,0,0,0,0,0,1,0,0`, `$skip=0,0,0,0,0,0,0,1,0`, `$skip=0,0,0,0,0,0,0,0,1` (skip number=1)
* Retain specific skip connections at different locations: take all the above combinations when skip number=1