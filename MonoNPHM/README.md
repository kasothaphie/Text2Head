# NPHM-TUM

This repository gives an implementation of an NPHM model with backward deformations, e.g. see [MonoNPHM](https://simongiebenhain.github.io/MonoNPHM/).


## Installation

```
conda env create -f environment.yml   
conda activate NPHM-TUM

# Install pytorch with CUDA support
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install PytorchGeometry and helper packages with CUDA support
conda install pyg -c pyg
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# Install Pytorch3D with CUDA support
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d=0.7.4 -c pytorch3d
```

Additionally, you will need to install the [famudy-data](https://github.com/tobias-kirschstein/famudy-data) package.
Make sure to follow the corresponding README.md

Before you can start to experiment with the provided demo scripts, you will need to set a few crucial paths in `src/nphm_tum/env_paths.py`:
- The `ASSETS` variable should point to `/rhome/sgiebenhain/NPHM/assets/`, whereever it is mounted.
- The `NERSEMBLE_DATASET_PATH` variable should point to `/cluster/doriath/tkirschstein/data/famudy/full/`
- The `EXPERIMENT_DIR_REMOTE` variable should point to `/cluster/doriath/sgiebenhain/GTA/experiments/`

Let me know if you run into any problems, or create a GitHub issue.

## Overview of the Available Data

For a quick overview of the dataset structure and how NPHM is registered against the frames of the multi-view dataset, refer to [dataset.md](https://github.com/SimonGiebenhain/NPHM-TUM/blob/58ea9db3c47e95ea53cd2ac530ac13363f3cc316/dataset.md)

## Getting Started

For some of the identites all timesteps of all sequences are processed. 
These subjects provide a good entry point to start your experiments with.
For example, take subject `037`.

You can find a script that demonstrates the usage of the NPHM model 
and reconstructs meshes from the fitted latent codes:

```
python scripts/intro_NPHM.py --p_id 37 --exp_name old2_imple_fabric_4gpu_all_traini2_wears --ckpt 6500 --seqs EXP-1-head
```

Importantly, the previous script only demonstrated the usage of the NPHM model as a "decoder".
Therefore, the resulting meshes are in canonical space.

Running

```
python scripts/alignment_NPHM.py --p_id 37 --seq_name EXP-1-head --frame 0
```
 demonstrates (the slightly convoluted way) of aligning the reconstructed mesh with world coordinates of the multi view camera system.
This is visualized in a 3D viewer, as well as, by rendering the mesh and comparing it against the camera image.
