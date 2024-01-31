import traceback

import tyro
from PIL import Image
import pymeshlab
from time import time

from nphm_tum.utils.reconstruction import create_grid_points_from_bounds, mesh_from_logits
from nphm_tum.models.reconstruction import get_logits, get_canonical_vertices



from nphm_tum.models.neural3dmm import construct_n3dmm, load_checkpoint
from nphm_tum import env_paths


import numpy as np
import json, yaml
import os
import os.path as osp
import torch
import pyvista as pv
import trimesh

os.umask(0o002)


def timeit(t0, tag):
    t = time()
    print(tag, t-t0)
    return t


def sampling(n3dmm, lat_codes, res=256):

    out_dir_rnd_samples = f'~/nphm_random_samples/'
    os.makedirs(out_dir_rnd_samples, exist_ok=True)


    std_geo = lat_codes.codebook['geo'].embedding.weight.std(dim=0)
    std_exp = lat_codes.codebook['exp'].embedding.weight.std(dim=0)

    mini = [-.55, -.6, -.95]
    maxi = [0.55, 0.75, 0.4]
    grid_points = create_grid_points_from_bounds(mini, maxi, res)
    grid_points = torch.from_numpy(grid_points).to(0, dtype=torch.float)
    grid_points = torch.reshape(grid_points, (-1, 3)).unsqueeze(0).to(0)

    global_step = 0

    for step in range(0, 10):

            z_geo = torch.randn([1, 1,  lat_codes.codebook['geo'].embedding.weight.shape[1]]).cuda() * std_geo.unsqueeze(0)


            z_exp = (torch.randn(
                [1, 1, lat_codes.codebook['exp'].embedding.weight.shape[1]]).cuda()) * std_exp.unsqueeze(0)


            encoding_val = {'geo': z_geo,
                             'exp': z_exp,
                             }
            torch.cuda.empty_cache()
            logits_val = get_logits(decoder=n3dmm,
                                    encoding=encoding_val,
                                    grid_points=grid_points.clone(),
                                    nbatch_points=50000,

                                    )

            level_set = mesh_from_logits(logits_val, mini, maxi, res)

            level_set.export(f'{out_dir_rnd_samples}/id_{global_step:05d}.ply')
            global_step += 1



def do_mc_fast(neural_3dmm,
          p_id,
          frame,
          flame_fitting_dir: str,
          seq_name: str,
          speedup_cache = None,
          resolution: int = 256
          ):
    device = 'cuda'
    t0 = time()
    if speedup_cache is None:
        speedup_cache = {}

    for nphm_fitting_folder in env_paths.nphm_tracking_name_priorities:
        src_dir = env_paths.NERSEMBLE_DATASET_PATH + '/{:03d}/sequences/{}/annotations/tracking/{}/'.format(
        p_id, seq_name, nphm_fitting_folder)
        if os.path.exists(src_dir):
            break
    if not os.path.exists(src_dir):
        raise ValueError('Could not find NPHM tracking for participant {} and sequence {}'.format(p_id, seq_name))


    tgt_dir = f'~/test_nphm_tum_repo/{p_id}_{seq_name}'
    os.makedirs(tgt_dir, exist_ok=True)

    # define the extends of the bounding cuboid
    mini = [-.55, -.6, -.95]
    maxi = [0.55, 0.75, 0.4]
    grid_points = create_grid_points_from_bounds(mini, maxi, resolution)
    grid_points = torch.from_numpy(grid_points).to(device, dtype=torch.float)
    grid_points = torch.reshape(grid_points, (-1, 3)).unsqueeze(0).to(device)

    # load the reconstructed latent codes
    lat_rep_shape = torch.from_numpy(np.load(src_dir + f'/{frame:05d}_id_code.npz')['arr_0']).to(device).unsqueeze(
        0).unsqueeze(0)
    lat_rep_expr = torch.from_numpy(np.load(src_dir + f'/{frame:05d}_ex_code.npz')['arr_0']).to(device).unsqueeze(
        0).unsqueeze(0)

    torch.cuda.empty_cache()

    # compute SDF values for all voxel centers
    with torch.no_grad():
        sdfs, deformations = get_logits(
            decoder=neural_3dmm,
            encoding={'geo': lat_rep_shape, 'exp': lat_rep_expr},
            grid_points=grid_points.clone(),
            nbatch_points=150000 if not os.path.exists('/mnt/rohan/') else 25000,
            approx_verts=speedup_cache.get('cached_vertices', None),
            cached_sdfs=speedup_cache.get('cached_sdfs', None),
            cached_deformations=speedup_cache.get('cached_deformations', None),
            return_deformations=True
        )

    t0 = timeit(t0, 'TIMEIT: FillVolume')

    speedup_cache['cached_sdfs'] = sdfs

    # extract the zero level set using marching cubes
    mesh = mesh_from_logits(sdfs.copy(), mini, maxi, resolution)

    t0 = timeit(t0, 'TIMEIT: MarchingCubes')

    speedup_cache['cached_vertices'] = mesh.vertices

    with torch.no_grad():
        torch.cuda.empty_cache()
        encoding = {'geo': lat_rep_shape, 'exp': lat_rep_expr}
        approx_verts = torch.from_numpy(speedup_cache['cached_vertices']).unsqueeze(0).float().to(encoding['geo'].device)

        in_dict = {'queries': approx_verts, 'cond': encoding}
        if hasattr(neural_3dmm.id_model,
                   'mlp_pos') and neural_3dmm.id_model.mlp_pos is not None and 'anchors' not in in_dict:
            in_dict.update({'anchors': neural_3dmm.id_model.get_anchors(encoding['geo'])})
        out_ex = neural_3dmm.ex_model(in_dict)
        speedup_cache['cached_deformations'] = out_ex['offsets'].detach().cpu().numpy()

    t0 = timeit(t0, 'TIMEIT: CacheDeformations')

    print('SAVING', tgt_dir)

    #mesh_nphm = mesh.copy()
    ####mesh = apply_transform(mesh, sim_nphm2mvs)

    # save the mesh as .CTM to save storage space
    # CTM meshes can be open with meshlab
    ms = pymeshlab.MeshSet()
    m = pymeshlab.Mesh(mesh.vertices, mesh.faces)
    ms.add_mesh(m)
    ms.save_current_mesh(tgt_dir + '/{:05d}.CTM'.format(frame), lossless=True)
    ms.clear()
    # alternatively you can export thr trimesh as .ply

    #canonical_attributes = get_canonical_vertices(neural_3dmm,
    #                                              encoding={'geo': lat_rep_shape, 'exp': lat_rep_expr},
    #                                              # {'geo': lat_rep_shape, 'exp': lat_rep_expr},
    #                                              vertices=torch.from_numpy(mesh_nphm.vertices).float().to(device),
    #                                              nbatch_points=20000)
    # np.savez_compressed(tgt_dir + '/{:05d}_canonical_vertices.npz'.format(frame),
    #                    canonical_attributes)

    return speedup_cache


def do_mc(neural_3dmm,
          p_id,
          frame,
          flame_fitting_dir : str,
          seq_name : str,
          resolution : int = 256,
          speedup_cache = None, # only used in "do_mc_fast"
        ):

    device = 'cuda'
    t0 = time()

    for nphm_fitting_folder in env_paths.nphm_tracking_name_priorities:
        src_dir = env_paths.NERSEMBLE_DATASET_PATH + '/{:03d}/sequences/{}/annotations/tracking/{}/'.format(
            p_id, seq_name, nphm_fitting_folder)
        if os.path.exists(src_dir):
            break
    if not os.path.exists(src_dir):
        raise ValueError('Could not find NPHM tracking for participant {} and sequence {}'.format(p_id, seq_name))

    tgt_dir = f'~/test_nphm_tum_repo/{p_id}_{seq_name}'
    os.makedirs(tgt_dir, exist_ok=True)


    # define the extends of the bounding cuboid
    mini = [-.55, -.6, -.95]
    maxi = [0.55, 0.75, 0.4]
    grid_points = create_grid_points_from_bounds(mini, maxi, resolution)
    grid_points = torch.from_numpy(grid_points).to(device, dtype=torch.float)
    grid_points = torch.reshape(grid_points, (-1, 3)).unsqueeze(0).to(device)


    # load the reconstructed latent codes
    lat_rep_shape = torch.from_numpy(np.load(src_dir + f'/{frame:05d}_id_code.npz')['arr_0']).to(device).unsqueeze(0).unsqueeze(0)
    lat_rep_expr = torch.from_numpy(np.load(src_dir + f'/{frame:05d}_ex_code.npz')['arr_0']).to(device).unsqueeze(0).unsqueeze(0)


    torch.cuda.empty_cache()


    # compute SDF values for all voxel centers
    with torch.no_grad():
        sdfs = get_logits(
                            decoder=neural_3dmm,
                            encoding={'geo': lat_rep_shape, 'exp': lat_rep_expr},
                            grid_points=grid_points.clone(),
                            nbatch_points=25000, # increase for high reconstruction speed until out-of-cuda memory
                        )

    # extract the zero level set using marching cubes
    mesh = mesh_from_logits(sdfs.copy(), mini, maxi, resolution)


    # save the mesh as .CTM to save storage space
    # CTM meshes can be open with meshlab
    ms = pymeshlab.MeshSet()
    m = pymeshlab.Mesh(mesh.vertices, mesh.faces)
    ms.add_mesh(m)
    ms.save_current_mesh(tgt_dir + '/{:05d}.CTM'.format(frame), lossless=True)
    ms.clear()
    # alternatively you can export thr trimesh as .ply
    speedup_cache = None
    return speedup_cache



def run_reconstruction(neural_3dmm,
                       pid,
                       seqs : str = ''):

    # all sequences that can possible be available
    all_seqs = ['EMO-1-shout+laugh',
                'EMO-2-surprise+fear',
                'EMO-3-angry+sad',
                'EMO-4-disgust+happy',
                'EXP-1-head', 'EXP-2-eyes', 'EXP-3-cheeks+nose', 'EXP-4-lips', 'EXP-5-mouth',
                'EXP-6-tongue-1', 'EXP-7-tongue-2', 'EXP-8-jaw-1', 'EXP-9-jaw-2', 'FREE', 'SEN-01-cramp_small_danger',
                'SEN-02-same_phrase_thirty_times', 'SEN-03-pluck_bright_rose', 'SEN-04-two_plus_seven',
                'SEN-05-glow_eyes_sweet_girl', 'SEN-06-problems_wise_chief', 'SEN-07-fond_note_fried',
                'SEN-08-clothes_and_lodging', 'SEN-09-frown_events_bad', 'SEN-10-port_strong_smokey']

    # if there is a commera separated string of sequences provided only process these ones
    if len(seqs) > 0:
        seqs = seqs.split(',')
        _all_seqs = []
        for seq in all_seqs:
            for s in seqs:
                if seq.startswith(s):
                    _all_seqs.append(seq)
        all_seqs = _all_seqs

    for pid in [pid]:
        for tgt_seq in all_seqs:
            try:

                tracking_dir = env_paths.NERSEMBLE_DATASET_PATH + '/{:03d}/sequences/{}/annotations/tracking/'.format(pid, tgt_seq)
                speedup_cache = None

                distinctive_frames = list(range(0, 2000, 3))
                # process frames
                for frame in distinctive_frames:
                    try:
                        print('starting mc')

                        # reconstruct a single mesh
                        # for a much faster, but slightly more complicated version, you can use "do_mc_fast" intead of "do_mc"
                        speedup_cache = do_mc(
                                  neural_3dmm,
                                  pid,
                                  frame,
                                  tracking_dir,
                                  tgt_seq,
                                  resolution=256, # resolution for marching cubes
                                  speedup_cache=speedup_cache,
                                  )
                    except Exception as e:
                        traceback.print_exc()
                        speedup_cache = None

            except Exception as e_outer:
                pass


def main(p_id: int,
         exp_name : str,
         ckpt : int,
         seqs: str = '',
         ):

    # load cmodel onfig files
    weight_dir_shape = env_paths.EXPERIMENT_DIR_REMOTE + '/{}/'.format(exp_name)
    fname_shape = weight_dir_shape + 'configs.yaml'
    with open(fname_shape, 'r') as f:
        print('Loading config file from: ' + fname_shape)
        CFG = yaml.safe_load(f)
    print('###########################################################################')
    print('####################     Model Configs     #############################')
    print('###########################################################################')
    print(json.dumps(CFG, sort_keys=True, indent=4))

    # load participant IDs that were used for training
    fname_subject_index = f"{weight_dir_shape}/subject_train_index.json"
    with open(fname_subject_index, 'r') as f:
        print('Loading subject index: ' + fname_subject_index)
        subject_index = json.load(f)

    # load expression indices that were used for training
    fname_subject_index = f"{weight_dir_shape}/expression_train_index.json"
    with open(fname_subject_index, 'r') as f:
        print('Loading subject index: ' + fname_subject_index)
        expression_index = json.load(f)


    # construct the NPHM models and latent codebook
    device = torch.device("cuda")
    neural_3dmm, latent_codes = construct_n3dmm(
                                  cfg = CFG,
                                  modalities=['geo', 'exp'],
                                  n_latents=[len(subject_index), len(expression_index)],
                                  device=device,
                                  )

    # load checkpoint from trained NPHM model, including the latent codes
    ckpt_path = osp.join(weight_dir_shape, 'checkpoints/checkpoint_epoch_{}.tar'.format(ckpt))
    print('Loaded checkpoint from: {}'.format(ckpt_path))
    load_checkpoint(ckpt_path, neural_3dmm, latent_codes)



    sampling(neural_3dmm, latent_codes)

    run_reconstruction(neural_3dmm,
                       p_id,
                       seqs=seqs, )


if __name__ == '__main__':
    tyro.cli(main)




