# Deep Photo Scan
#### [[Page]](https://minhmanho.github.io/dpscan/) [[Paper]](https://arxiv.org/abs/2102.06120) [[SupDoc]](https://drive.google.com/file/d/1atoOvwUAnS7qJ438w8GB-mmwg5QwwLrf/view?usp=sharing) [[DEMO]](https://youtu.be/-kC2T6vMlpQ)

![Alt Text](https://raw.githubusercontent.com/minhmanho/dpscan/master/docs/images/dpscan.gif)

[Deep Photo Scan: Semi-supervised learning for dealing with the real-world degradation in smartphone photo scanning](https://arxiv.org/abs/2102.06120)<br>
[Man M. Ho](https://minhmanho.github.io/) and [Jinjia Zhou](https://www.zhou-lab.info/jinjia-zhou)<br>
In ArXiv, 2021.

## Prerequisites
- Ubuntu 16.04
- Pillow
- [PyTorch](https://pytorch.org/) >= 1.3.0
- Numpy
- gdown (for fetching models)

## Get Started
### 1. Clone this repo
```
git clone https://github.com/minhmanho/dpscan.git
cd dpscan
```

### 2. Fetch the pre-trained model
You can download the pre-trained model at [here](https://drive.google.com/uc?id=1LyMXV_wx_G3DMtKTV6BtjR22rlnweaB-) (148MB) or run the following script:

```
./models/fetch_model.sh
```

## Smartphone-scanned Photo Restoration
Run our semi-supervised Deep Photo Scan to restore smartphone-scanned photos as:

```
CUDA_VISIBLE_DEVICES=0 python run.py \
    --in_dir ./data/in/ \
    --out_dir ./data/out/ \
    --ckpt ./models/dpscan_saved_weights.pth.tar \
    --size 1072x720
```

Check [our page](https://minhmanho.github.io/dpscan/) for the results.

## DIV2K-SCAN dataset
Training data captured using iPhone XR can be downloaded at this [Google Drive](https://drive.google.com/file/d/1n6G1MzLBLKLj3zDpj1ziObq8gsG8wzKo/view?usp=sharing).
![Alt Text](https://raw.githubusercontent.com/minhmanho/dpscan/master/docs/images/train.jpg)

Besides photos in the same distribution as training photos, test data also consists of out-of-distribution cases such as color-balanced and taken-by-XperiaXZ1 photos. All test cases can be downloaded at this [Google Drive](https://drive.google.com/file/d/1atyzBBLWNOQCdzPIkmfD3h6OlEU2tNTi/view?usp=sharing).
![Alt Text](https://raw.githubusercontent.com/minhmanho/dpscan/master/docs/images/test.jpg)

## Citation
If you find this work useful, please consider citing:
```
@misc{ho2021deep,
    title={Deep Photo Scan: Semi-supervised learning for dealing with the real-world degradation in smartphone photo scanning},
    author={Man M. Ho and Jinjia Zhou},
    year={2021},
    eprint={2102.06120},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Acknowledgements
We would like to thank:
- _digantamisra98_ for the [Unofficial PyTorch Implementation of EvoNorm ](https://github.com/digantamisra98/EvoNorm)
```
Liu, Hanxiao, Andrew Brock, Karen Simonyan, and Quoc V. Le. "Evolving Normalization-Activation Layers." 
ArXiv (2020).
```
- _Richard Zhang_ for the [BlurPool](https://github.com/adobe/antialiased-cnns/blob/master/antialiased_cnns/blurpool.py)
```
Zhang, Richard. "Making convolutional networks shift-invariant again." 
ICML (2019).
```
- Timofte et al. for the [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
```
Timofte, Radu, Shuhang Gu, Jiqing Wu, and Luc Van Gool.
"Ntire 2018 challenge on single image super-resolution: Methods and results."
CVPR Workshops (2018).
```
## License
This work, including the trained model, code, and dataset, is for **non-commercial** uses and research purposes only.

## Contact
If you have any questions, feel free to contact me (maintainer) at [manminhho.cs@gmail.com](mailto:manminhho.cs@gmail.com)
