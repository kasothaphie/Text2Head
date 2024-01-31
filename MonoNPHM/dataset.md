
## Registration Procedure

The registration/alignment of NPHM against the sequences of the multi-view dataset contains the following steps:
1. Take the identity code `z_id` from the training dataset consisting of complete 3D scans.
2. Fit the `FLAME` against the `COLMAP` point cloud. This gives us an estimate of the rigid pose of the head within the world coordinates.
3. Optimize for `z_ex` as well as a corrective, rigid transform `[R|t]`, by minimizing distance between the COLMAP point cloud and the surface predicted by the NPHM model.

# Dataset Structure

The dataset root can be found at `DROOT='/cluster/doriath/tkirschstein/data/famudy/full'` and is structured as follows:

All data of one participant, who is identified by a unique 3-digit number `PARTICIPANT_ID`, 
is contained in `DROOT/{PARTICIPANT_ID}/`. 

These folders are further structured into the contents of the individual sequences, that were captured, i.e.

`DROOT/{PARTICIPANT_ID}/sequences/{SEQUENCE_NAME}`

holds all data of one sequence, where `SEQUENCE_NAME` either starts with:
- `EXP`, denoting sequences with specific facial expressions. There are 9 different expression sequences.
- `EMO`, denoting sequences composed of two distinct emotions. There are 4 different emotion sequences.
- `SEN`, denoting talking sequences. There are 10 differeent sentences.
- `FREE`, denoting a special sequence, where the participants do what they want.

### The `Timesteps` Folder
Each sequence folder has information stored per timestep
in 
`DROOT/{PARTICIPANT_ID}/sequences/{SEQUENCE_NAME}/timesteps/frame_XXXXX/`

Note: Only every third frame is processed, since we recorded with 71 frames per second.

The `timesteps` folder should have the images of all 16 cameras stored in `images-2x` (indicating the resolution was reduced by a factor of 2).

The images are named in the convention `cam_{SERIAL}.jpg`, where `SERIAL` is a unique identifier for each camera that is shared throughout the whole dataset.

Furthermore, there are depth maps, normal maps and point clouds from COLMAP provided, as well as, semantic segmentations, and background segmentation maps.

### The `Annotations` Folder

Next to the `timestep` folder there is an `annotations` folder.
The annotation folder has information that is stored in one chunk per sequence, compared to per-timestep.
Additionally, the annotation folder contains the tracking and registrations results.

The following information is available as a single `.npy` file per sequence and camera:
- `bbox/cam_{SERIAL}.npy` has shape `[n_processed_frames x 5]`. The first 4 dimensions encode the location and size of the 2D bounding box (between 0 and 1, i.e. realtive to the image size). The last dimensions holds a confidence value for the detected face.
- `landmarks2D/PIPnet/{SERIAL}.npy` holds the detections of the PIPnet landmark detector. It uses WFLW landmarks and therefore has a shape of `[n_processed_frames x 98 x 2]`, i.e. there are 98 landmarks which are also stored relative to the image size.
- `color_correction/{SERIAL}.npy` holds information of an affine color transformation matrix.

#### FLAME tracking

`tracking/FLAME2023_v2` contains tracked FLAME parameters, using the FLAME 2023 model (with jaw rotation). 
More informtion to come.

#### NPHM tracking

The fitted expression codes can be found in

`tracking/NPHM_{APPENDIX}/`, where `APPENDIX` can be one of `v2`, `_distinctive`, `_temp`, ...
Sorry for the mess. I hope to clean it up in the following weeks.
In all these cases, the following quantities are available:

- identity codes `z_id` are available in `XXXX_id_code.npz`
- expression codes `z_ex` are available in `XXXX_ex_code.npz`

where `XXXXX` denotes the zero-padded timestep, e.g. `00012`.

The `Getting Started` section gives further instructions on how to use the NPHM codes and how to align the meshes with the cameras.



## The `famudy-data` Package

Tobias, has written a small python package, that manages the data loading of images, point clouds, segmentation masks, camera parameters etc.
We highly suggest to use that package, such that you don't have to deal with all the paths etc.

Have a look at the `README.md` in the `famudy-data` GitHub page for installation and usage instructions.

The repository allows to specify any `DROOT`, i.e. it can also be used outside of our cluster, but the data would have to follow the same folder structure.
Thus far the GitHub repository is private, if you tell us your `GitHub user name` Tobias can add you as a collaborator.


## PROCESSING STATUS

Since processing the whole dataset is a tremendous computation burden, 
only a few random sequences per subject (usually 3) have the COLMAP point clouds available.

For now, I did not have the need for registereding every available frame of every available sequence (there are a few exceptions: 18, 37, 38, 124, 251, 264). 
Instead, I devised a simple heuristic to select a subset of the 20 most distinct frames for each available sequence.

The selected frames can be found in `DROOT/{PARTICIPANT_ID}/distinctive_frames/frames_{SEQUENCE_NAME}.npy`, 
which holds the 20 selected frames indices. 
Additionally, `DROOT/{PARTICIPANT_ID}/distinctive_frames/frames_{SEQUENCE_NAME}.jpg` shows the images from the frontal camera of the selected frames. 
Note, that the very first frame is append as the first frame, making it 21 frames in total.

