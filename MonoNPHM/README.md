# NPHM-TUM

This repository gives an implementation of an NPHM model with backward deformations, e.g. see [MonoNPHM](https://simongiebenhain.github.io/MonoNPHM/).

The currently this repository focuses on single image 3D face reconstruction using inverse rendering.

## Installation
> Note that some of the steps below can take a while
```
conda env create -f environment.yml   
conda activate NPHM-TUM

pip install -e .

# Install pytorch with CUDA support
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install PytorchGeometry and helper packages with CUDA support
conda install pyg -c pyg
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# Install Pytorch3D with CUDA support
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d=0.7.4 -c pytorch3d
```

Next, you need to fill in some paths in `./src/nphm_tum/env_paths.py`.
Before this use `cp src/nphm_tum/env_paths.template src/nphm_tum/env_paths.py` to create your local version and 
set the paths according to your system.
The provided comments are hopefully enough explanation.

Finally, fix some numpy versioning:
`pip install numpy==1.23`

## Getting Started with MonoNPHM

The following gives intructions how to run the single-image 3D reconstruction pipeline.

### Installing the Preprocessing Pipeline

Our preprocessing pipeline relies on the FLAME model. Therefore, you will need an account for the [FLAME website](https://flame.is.tue.mpg.de/).
Let me know if you have any trouble concerning that.

Also, you will need to download the pretrained [normal detector](https://github.com/boukhayma/face_normals/tree/5d6f21098b60dd5b43f82525383b2697df6e712b) from [here](https://drive.google.com/file/d/1Qb7CZbM13Zpksa30ywjXEEHHDcVWHju_/edit).
Place the downloaded `model.pth` into `src/nphm_tum/preprocessing/pretrained_models/model.pth`.

Similarly, download the weights for the employed [facial landmark detector](https://github.com/jhb86253817/PIPNet) from [here](https://drive.google.com/drive/folders/17OwDgJUfuc5_ymQ3QruD8pUnh5zHreP2).
Download the folder `snapshots/WFLW` and place it into `src/nphm_tum/preprocessing/PIPnet/snapshots`. 

### Download Pretrained Model, Assets and Example Data

You can find all necessary data here: `https://drive.google.com/drive/folders/1yZdQkkKwBJLeMIsCSAy7MeAkfJjZVD_H?usp=sharing`
Please don't carelessly share the model checkpoint, since the latent codes from the dataset can reconstruct the faces up to a high level of detail. 

### Running the Preprocessing

First we need to preprocess a bunch of data, namely this includes:
- landmark detection
- semantic segmentation (including forground background segmentation)
- FLAME fitting to get a rough initialization of the camera pose
- Face Normal prediction

To run all necessary preprocessing steps for id `00995` run:


```
cd scripts/preprocessing
./run.sh 00995
cd ../..
```

### Running the Tracking

Once the preprocessing is done, you can start the inverse rendering using:

```commandline
python scripts/inference/track.py --exp_name PretrainedMonoNPHM --ckpt 2500 --seq_tag 00995 --run_tag first_try 
```

> Note: there is also an option to enable a normal loss. But the normal detector is not quite that reliable.

