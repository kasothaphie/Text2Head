# Installation
MAYBE NEED __RECURSE SUBMODULE

incase you get an error bc. of np.int, check which numpy version is actually used. I had a case where two numpy version were installed and it was using 1.26 instead of 1.23

### Matting
- Download [modnet_webcam_portrait_matting.ckpt](https://drive.google.com/file/d/1Nf1ZxeJZJL8Qx9KadcYYyEmmlKhTADxX/view?usp=sharing) and put it into `src/nphm_tum/preprocessing/MODNet/pretrained/`
```commandline
cd ./src/nphm_tum/preprocessing
git clone git@github.com:ZHKKKe/MODNet.git
git clone git@github.com:jhb86253817/PIPNet.git
git clone git@github.com:Zielon/MICA.git
git clone git@github.com:Zielon/metrical-tracker.git
```
### Face and Landmark Detection

#### Face Detection
```
cd src/nphm_tum/preprocessing/PIPNet/FaceBoxesV2/utils/
python build.py build_ext --inplace
```

#### Landmark Detection
```commandline
cd ../..
mkdir snapshots
```

Download the `WFLW` from [here](https://drive.google.com/drive/folders/1fz6UQR2TjGvQr4birwqVXusPp6tMAXxq).


#### MICA

You will need to have an Account for the [FLAME model] (https://flame.is.tue.mpg.de/index.html). 
Your account name and password are need during installation to download the FLAME model weights.

```commandline
cd src/nphm_tum/preprocessing/MICA
./install.sh
```

#### metrical tracker
```commandline
cd src/nphm_tum/preprocessing/metrical-tracker
./install.sh
```

> in config file should adopt fps properly!

replace generate_dataset.py
replace tracker.py
repace configs.py (needed to add `intrinsics_provided` field)