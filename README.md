# DANGER: A Danger Awareness Neural Generative Extension Framework for Robustness test
We have released the DANGER Dataset publicly to aid the research community in making advancements in model robustness and explainability.

The DANGER framework is universal to many opensource dataset. In this repo we provided two datasets - the vKITTI1 add-on dataset and the vKITTI2 add-on dataset.

## Overview
- [Changelog](#changelog)
- [Model Zoo](#model-zoo)
- [Tutorials](#tutorials)
- [Citation](#citation)

## Changelog
[2021-06-09] `DANGER` v0.1.0 is released.

## Introduction

### What does `DANGER` do?

`0002` Cut-in Opposite

![Origin](https://github.com/jayhsu0627/DANGER/blob/main/Pics/0002_clone.gif)
![Cut-in](https://github.com/jayhsu0627/DANGER/blob/main/Pics/0002_motion.gif)


`0018` Cut-in

![Origin](https://github.com/jayhsu0627/DANGER/blob/main/Pics/0018_clone.gif)
![Cut-in](https://github.com/jayhsu0627/DANGER/blob/main/Pics/0018_motion.gif)

`0006` Lane Change

![Origin](https://github.com/jayhsu0627/DANGER/blob/main/Pics/0006_clone.gif)
![Lane Change](https://github.com/jayhsu0627/DANGER/blob/main/Pics/0006_motion.gif)


### `DANGER` design pattern


## Model Zoo

### vKITTI1 
## Prerequisites

- Linux
- Python 3.6+
- PyTorch 0.4
- NVIDIA GPU (GPU memory > 8GB) + CUDA 9.0
- Conda
  
Use our notebook hosted on Colab/Colab Pro will help you setup all the prerequisites automatically.

```
Sun Jun  5 01:15:01 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla P100-PCIE...  Off  | 00000000:00:04.0 Off |                    0 |
| N/A   37C    P0    28W / 250W |      0MiB / 16280MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
Python 3.6.10 :: Anaconda, Inc.
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2017 NVIDIA Corporation
Built on Fri_Sep__1_21:08:03_CDT_2017
Cuda compilation tools, release 9.0, V9.0.176
time: 2min 19s (started: 2022-06-05 01:12:42 +00:00)
```

## Tutorials
We host our tutorial notebooks on Colab, and we suggest you initialize your vm with a GPU when you setup the 3D-SDN environment.

```
sudo chmod 755 ./DANGER/install.sh
./DANGER/install.sh
```
1 [3D-SDN environment](https://github.com/jayhsu0627/DANGER/blob/main/docs/CUDA9_3D_SDN.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zOJ8YydPx3a-cLl8nwKA7M66eJ5DhlSU?usp=sharing)
>3D-SDN environment:
* There are sections in this notebook: `Hardware Setup`, `3D-SDN Example`, `Semantic Training`, `Geometry Training`, `Textural Training`, and `test.json`
* Please run `Hardware Setup` to modify the default supported drivers that Colab provided.
* Run `3D-SDN Example` to verify the installation, which would help you finish the `Getting Started` section described in 3D-SDN.
* Use `test.json`, provided in step 2, to generate your dataset. You can ignore the `Semantic Training`, `Geometry Training`, `Textural Training` steps, but it would be helpful when you interested to train you own model.

2 [JSON generator](https://github.com/jayhsu0627/DANGER/blob/main/docs/Design_JSON.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vFl4zFHlEVxBBqOe_Q6wgT_zsQC1F8W_?usp=sharing)

Generated `JSON` files are preserved at [JSONs](/JSONs/).

## License
`DANGER` is released under the [MIT License](/LICENSE)

## Acknowledgement
`DANGER` is an open source project for dataset manipulation and model robustness test.
We would like to thank the authors of `3D-SDN` for their open-source release.

## Citation
If you find this project useful in your research, please consider citing:

```
@article{DANGER,
    title={},
    author={A and B},
    journal = {Neurips},
    year={2022}
}

```
## GitHub Push Code
```
git add .

git commit -m 'init'

git branch -M main

git push -u origin main
```
## To Do

- [x] Calculate `u,v` 
- [ ] Verify `u,v`  by check each operation in `vkitti_edit_benchmark.json`
- [ ] Try generate `*.json` by generate `delete` operation
- [ ] Try generate `*.json` by generate `ry` operation
- [ ] Try generate `*.json` by change `u,v` operation


