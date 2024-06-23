# Text2Head — CLIP-guided Latent Code Optimization
[Paper](./extra/report.pdf)

![heads](https://github.com/kasothaphie/Text2Head/blob/mono_nphm_AdamW/extra/heads.png)

This repository contains the source code for the paper **Text2Head — CLIP-guided Latent Code Optimization**.

## Abstract
We propose Text2Head, a novel method for generating neural parametric 3D head models driven by text descriptions. Our approach takes textual prompts describing a person and outputs latent codes for geometry and appearance, which are then used to generate textured 3D head geometries with a pre-trained Monocular Parametric Head Model (MonoNPHM). In contrast to existing approaches, we do not require the prior generation of ground truth pairs of text prompts and latent codes, which can be limited in quality and availability. Instead, our method allows direct optimization of latent codes leveraging a CLIP loss. Our method demonstrates the capability to faithfully generate 3D head models for various applications.

## Table of Contents
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgement](#acknowledgement)

## Setup
The full Setup is not possible at the moment, as the project is heavily based on on MonoNPHM which is not public yet.
For the rest follow this tutorial.

Create an environment via 
```
conda create -n "Text2Head" python=3.9
conda activate Text2Head
```
and install everything required with
```
pip install -r requirements.txt
```

Once MonoNPHM is public to follow their instructions to load weights and install all necessary packages.

## Usage
`notebooks/usage.ipynb` provides a script for generating scenes using the trained denoising network.

## Results
The figure below provides the results for given input text prompts.
![results](https://github.com/kasothaphie/Text2Head/blob/mono_nphm_AdamW/extra/qualitative.png)

## Acknowledgement
This work is developed with [TUM Visual Computing Group](http://niessnerlab.org) led by [Prof. Matthias Niessner](https://niessnerlab.org/members/matthias_niessner/profile.html). We thank Matthias for his great support and supervision.
Our work builds upon [MonoNPHM](https://arxiv.org/abs/2312.06740) by [Simon Giebenhain](https://simongiebenhain.github.io).
