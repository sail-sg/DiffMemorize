# On Memorization in Diffusion Models

## Environments

* We run all our experiments on A100 GPUs

* 64-bit Python 3.8 and PyTorch 1.13.

* Python libraries: See [environment.yml](./environment.yml) for exact library dependencies. You can use the following commands with Miniconda3 to create and activate your Python environment:
  - `conda env create -f environment.yml -n edm`
  - `conda activate edm`

## Datasets

We run our experiments on the CIFAR-10 and ImageNet datasets.

CIFAR-10 can be downloaded by the following commands:
```
mkdir datasets
mkdir datasets/cifar10
wget -P datasets/cifar10 https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```

To download ImageNet, please refer to [ImageNet Object Localization Challenge](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) and save it to `datasets/imagenet`. 


## Running Experiments
To generate images in Figure 1 (a) and (b), run following commands on one A100 GPU:
```
bash scripts/run_fig1_ab.sh
```

Our following experiments are run on 8 A100 GPUs, our provided code is through DDP with multi-node training, whose format is
```
torchrun --nproc_per_node 1 \
         --nnodes $WORLD_SIZE \
         --node_rank $RANK \
         --master_addr $MASTER_ADDR \
         --master_port $MASTER_PORT \
         train.py --parse=$parse
```

Alternatively, you can change it as follows to supprot DDP with single-node training
```
torchrun --standalone --nproc_per_node=8 train.py --parse=$parse
```
### Motivations
1. To reproduce our experiments in Figure 1(c), here is an example
```
bash scripts/run_fig1_cd.sh 10000 fig1_c/size10000 200
```

2. To reproduce our experiments in Figure 1(d), here is an example
```
bash scripts/run_fig1_cd.sh 1000 fig1_d/size1000 2000
```

### Data distribution
To reproduce our experiments of data dimension in Figure 2(a), here is an example
```
bash scripts/run_fig2_dim.sh 5000 32 fig2_a/size5000_res32 16
bash scripts/run_fig2_dim.sh 5000 16 fig2_a/size5000_res16 8
bash scripts/run_fig2_dim.sh 5000 8 fig2_a/size5000_res8 4
```

To reproduce our experiments of data inter-diversity in Figure 2(b), here is an example
```
bash scripts/run_fig2_inter.sh 2000 5 fig2_b/size2000_class5
```

To reproduce our experiments of data intra-diversity in Figure 2(c), here is an example
```
bash scripts/run_fig2_intra.sh 2000 0.5 fig2_c/size2000_alpha0.5
```


### Model configuration: model width/depth/time embedding
To reproduce our experiments in Figure 3, here is an example, please change the arguments to reproduce all experiments
```
bash scripts/run_fig3_model.sh 2000 fig3/size2000_2_128_positional 2 128 positional
```
or 

```
bash scripts/run_fig3_model.sh 1000 fig3/size1000_4_128_fourier 4 128 fourier
```

### Model configuration: skip connections
To reproduce our experiments in Figure 4, here is an example for DDPM++
```
bash scripts/run_fig4_skip.sh 1000 fig4/size1000_ddpmpp_skip1,1,1,1,1,1,1,1,1 vp ddpmpp positional 1,1,1,1,1,1,1,1,1
```

Here is another example for NCSN++
```
bash scripts/run_fig4_skip.sh 1000 fig4/size1000_ncsnpp_skip0,1,1,1,1,1,1,1,1 ve ncsnpp fourier 0,1,1,1,1,1,1,1,1
```

### Training procedure
To reproduce our experiments in Table 1, here is an example
```
bash scripts/run_table1_2.sh 2000 tbl1/size2000_batch256 256 1e-4 0.0
```

To reproduce our experiments in Table 2, here is an example
```
bash scripts/run_table1_2.sh 2000 tbl2/size2000_decay1e-4 512 2e-4 1e-4
```

To reproduce our experiments in Table 3, here is an example
```
bash scripts/run_table3.sh 2000 tbl3/size2000_ema0.999 0.999
```

### Unconditional v.s. Conditional
To reproduce our experiments for conditional diffusion models with true labels
```
bash scripts/run_fig5_cond.sh 2000 fig5/size2000_true_cond
```

To reproduce our experiments for conditional diffusion models with random labels
```
bash scripts/run_fig5_random.sh 2000 50 fig5/size2000_random_cond50
```

To reproduce our experiments for conditional diffusion models with unique labels
```
bash scripts/run_fig5_unique.sh 2000 fig5/size2000_unique
```

To reproduce our experiments for conditional EDM with unique labels
```
bash scripts/run_fig5_edm.sh fig5/size50k_edm_unique
```