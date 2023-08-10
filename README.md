<p align="center">
<a href="https://www.python.org/downloads/">
        <img alt="Build" src="https://img.shields.io/badge/3.9-Python-green">
</a>
<a href="https://pytorch.org">
        <img alt="Build" src="https://img.shields.io/badge/1.13-PyTorch-green">
</a>
<a href="https://github.com/NVlabs/stylegan3">
        <img alt="Build" src="https://img.shields.io/badge/NVlabs-Stylegan3-green">
</a>
<br>
<a href="https://arxiv.org/pdf/2304.07361.pdf">
            <img alt="Build" src="https://img.shields.io/badge/arXiv-2304.07361-blue">
    </a>
</p>

<h1 align="center">
    <p>Image Generator Watermarking</p>
</h1>


This repository is the official implementation of our USENIX'23 paper "PTW: Pivotal Tuning Watermarking for 
Pre-Trained Image Generators".
This source code makes it possible to watermark _any_ pre-trained image generator (GANs, Diffusion, ..) using few computational resources. The provided code only supports watermarking GANs, but we provide watermarking keys
which also allow watermarking diffusion models through Pivotal Tuning.
**Watermarking requires no training data!**

by [Nils Lukas](https://nilslukas.github.io/) and [Florian Kerschbaum](https://cs.uwaterloo.ca/~fkerschb/). 

If you find our code or paper useful, please cite
```
@InProceedings{lukas2023ptw,
  title      = {PTW: Pivotal Tuning Watermarking for Pre-Trained Image Generators},
  author     = {Lukas, Nils and Kerschbaum, Florian},
  journal    = {32nd USENIX Security Symposium},
  year       = {2023},
  url        = {https://arxiv.org/pdf/2304.07361.pdf}
}
```

## Description

This repository allows (i) training watermarking decoders and (ii) watermarking image generators using pre-trained decoders.
We include configuration files for the [StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch) and [StyleGAN3](https://nvlabs.github.io/stylegan3/) generators trained on the FFHQ dataset
in the `configs` folder. The `configs` folder also contains the configuration files for the watermarking decoders.

**Pre-Trained Watermarking Keys**

You can train your own keys any dataset by adjusting the configuration files. We provide
two pre-trained keys trained on a facial image dataset (FFHQ). 
* `pretrained_models/ptw-key-40-bit-ffhq-256-resnet50.pt`: A watermarking key trained on the StyleGAN2 generator trained on the FFHQ dataset. [link](https://www.dropbox.com/scl/fi/yrej1hli3kn8b6xmdx8nj/ptw-key-40-bit-ffhq-256-resnet50.pt?rlkey=y5ydv0krmehoaje6ecp8d3dvz&dl=0)

**Configuration Files**  

* `assets/pre_trained_urls.txt`: Contains public, pre-trained checkpoints of GANs hosted by NVIDIA and others. 
* `configs/keygen`: Contains parameters for the key generation (i.e., training the watermarking decoders). Training a decoder can take several GPU hours
depending on your setup and **there is no need to train your own watermarking key**, since we provide pre-trained checkpoints for FFHQ (see below).
* `configs/embed`: Contains parameters for embedding a watermark using the Pivotal Tuning method. While it is possible to pair any watermarking key with any 
generator, we suggest using a watermarking key trained on the generator's domain. 

**Command Line Interface (CLI)**  
* `examples/keygen.py`: Trains a watermarking decoder using the method presented in the paper.
* `examples/embed.py`: Embeds a watermark into a pre-trained generator using the Pivotal Tuning method.
* `examples/generate_images.py`: Generate images using a pre-trained generator and saves them to disk.
* `examples/detect.py`: Given (i) a watermarking key and (ii) a folder containing generated images, extract a watermarking message from each image
and compute the mean bit-accuracy. 


## Build & Run

This repository is compatible with the [StyleGAN-Xl](https://github.com/autonomousvision/stylegan-xl) implementation and has the same requirements.
Please refer to their repository when the instructions below fail.
```shell
$ conda env create -f environment.yml
$ conda activate ptw
$ pip install -e . 
```

**Pre-Trained Models**

Create a folder `pretrained_models` and download the following files:
* [`model_ir_se50.pth`](https://www.dropbox.com/s/abk2q3glwa0k43y/model_ir_se50.pth?dl=0) (_175 MB_): A model to compare facial identities. 

## Example Usage
1.) Train a watermarking key from scratch. Creates a watermarking key `pretrained_models/ptw-key-40-bit-ffhq-256-2.pt`.
**Note**: *You can also skip this step and download one of our
pre-trained watermarking keys.* 
```shell 
python keygen.py --config_path ../configs/keygen/ffhq256/keygen-sg2-ffhq256.yml
```
2.) Embed a watermark into a pre-trained StyleGAN2 generator. Creates a watermarked model `pretrained_models/ptw-generator-40-bit-ffhq-256-2.pt`.
```shell
python embed.py --config_path ../configs/embed/ffhq256/ptw-sg2.yml
```
3.) Generate watermarked images. Creates a folder `generated_images/ptw-generator-40-bit-ffhq-256-2/` with 1k images.
```shell
python generate_images.py --config_path ../configs/generate_images/ffhq256/ptw-sg2.yml
```
4.) Verify watermarked images. Outputs statistics on the bit-accuracy of the watermark for each image. 
```shell
python detect.py --config_path ../configs/detect/ffhq256/ptw-sg2.yml
```

## Logging 

We support wandb logging. To enable it, please set `env_args->logging_tool: wandb` in the configuration file.
  



