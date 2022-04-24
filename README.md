# Automatic-Background-Extraction

Automatic Background Extraction combines semantic segmentation and inpainting models to remove vehicles from images taken from a satellite view. 

1. Semantic Segmentation is used to assign class identifications to each pixel in the input image.
2. Identified vehicles are replaced with white pixels and a corresponding binary white-on-black mask is generated
3. The altered image and mask is passed to an inpainting model to fill in the degradation generated in the previous step

The SegNet model (https://arxiv.org/abs/1511.00561) has been combined with EdgeConnect (https://github.com/knazeri/edge-connect) to produce this project. NVIDIA CUDA / cuDNN, and CPU are supported.

## Requirements
* Python 3.8
* PyTorch 1.10.x

## Installation
```
git clone https://github.com/StanleyLiu17/automatic-background-extraction.git
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

Install pre-trained weights for SegNet based on the iSAID dataset and move them into the checkpoints folder: https://drive.google.com/drive/folders/168E5Sju0VEGEXIJkFlxHhWcIieEpeZFL?usp=sharing

The iSAID dataset can be found here: https://captain-whu.github.io/iSAID/

Install pre-trained weights for EdgeConnect based on the places2 dataset and move them into the checkpoints folder: https://drive.google.com/drive/folders/1KyXz4W4SAvfsGh3NJ7XgdOv5t46o-8aa
(NOTE: This link will be replaced with custom trained weights when training is complete)

We use the iSAID dataset and qd-imd irregular mask dataset. The latter can be found here: https://github.com/karfly/qd-imd

## Inference
```python test.py --input [PATH_TO_INPUT_DIR] --output [PATH_TO_OUTPUT_DIR]```
The GPU is used by default! If you want to use CPU add the ```--cpu``` flag to the above command

Note: The inpainting model, EdgeConnect, will output 256 x 256 images by default. This project implements automatic image slicing and stitching, so result images are always in the same resolution as input images, though it uses more memory at runtime.

## Training
For training EdgeConnect, please refer to the repo here: https://github.com/knazeri/edge-connect
For training SegNet, please refer to the repo here: https://github.com/nshaud/DeepNetsForEO, or use the provided ```train_segnet.ipynb``` notebook adapted from the aforementioned listed repo for the express purpose of training SegNet.

## To-Dos
* Optimize SegNet training
* Complete inpainting dataset by manually inpainting and augmentation of the rest of the iSAID training, validation, and testing datasets
* Complete EdgeConnect training and optimize parameters
* Optimize as needed based on the collected datasets of areas of interest
* Various minor optimizations

## EdgeConnect License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International.](https://creativecommons.org/licenses/by-nc/4.0/)

Except where otherwise noted, this content is published under a [CC BY-NC](https://github.com/knazeri/edge-connect) license, which means that you can copy, remix, transform and build upon the content as long as you do not use the material for commercial purposes and give appropriate credit and provide a link to the license.

## SegNet License
RESEARCH AND NON COMMERCIAL PURPOSES
Code license
For research and non commercial purposes, all the code and documentation is released under the GPLv3 license:

Copyright (c) 2017 ONERA and IRISA, Nicolas Audebert, Bertrand Le Saux, Sébastien Lefèvre.

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

PLEASE ACKNOWLEDGE THE ORIGINAL AUTHORS AND PUBLICATION ACCORDING TO THE REPOSITORY github.com/nshaud/DeepNetsForEO OR IF NOT AVAILABLE: Nicolas Audebert, Bertrand Le Saux and Sébastien Lefèvre "Semantic Segmentation of Earth Observation Data Using Multimodal and Multi-scale Deep Networks", Asian Conference on Computer Vision, 2016.

## Citations

```
inproceedings{nazeri2019edgeconnect,
  title={EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning},
  author={Nazeri, Kamyar and Ng, Eric and Joseph, Tony and Qureshi, Faisal and Ebrahimi, Mehran},
  journal={arXiv preprint},
  year={2019},
}

@InProceedings{Nazeri_2019_ICCV,
  title = {EdgeConnect: Structure Guided Image Inpainting using Edge Prediction},
  author = {Nazeri, Kamyar and Ng, Eric and Joseph, Tony and Qureshi, Faisal and Ebrahimi, Mehran},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV) Workshops},
  month = {Oct},
  year = {2019}
}

@article{audebert_beyond_2017,
title = "Beyond RGB: Very high resolution urban remote sensing with multimodal deep networks",
journal = "ISPRS Journal of Photogrammetry and Remote Sensing",
year = "2017",
issn = "0924-2716",
doi = "https://doi.org/10.1016/j.isprsjprs.2017.11.011",
author = "Nicolas Audebert and Bertrand Le Saux and Sébastien Lefèvre",
keywords = "Deep learning, Remote sensing, Semantic mapping, Data fusion"
}

@inproceedings{waqas2019isaid,
title={iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images},
author={Waqas Zamir, Syed and Arora, Aditya and Gupta, Akshita and Khan, Salman and Sun, Guolei and Shahbaz Khan, Fahad and Zhu, Fan and Shao, Ling and Xia, Gui-Song and Bai, Xiang},
booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
pages={28--37},
year={2019}
}

@article{DBLP:journals/corr/BadrinarayananK15,
  author    = {Vijay Badrinarayanan and
               Alex Kendall and
               Roberto Cipolla},
  title     = {SegNet: {A} Deep Convolutional Encoder-Decoder Architecture for Image
               Segmentation},
  journal   = {CoRR},
  volume    = {abs/1511.00561},
  year      = {2015},
  url       = {http://arxiv.org/abs/1511.00561},
  eprinttype = {arXiv},
  eprint    = {1511.00561},
  timestamp = {Mon, 13 Aug 2018 16:46:06 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/BadrinarayananK15.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
}
