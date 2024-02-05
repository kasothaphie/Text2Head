import traceback

import json, os, yaml
from os import path as osp
import torch
import numpy as np
import tyro
import random
from typing import Optional

from nphm_tum.fitting.photometric_tracking import TrackerMonoNPHM
from nphm_tum.fitting.wrapper import MonoNPHM

from nphm_tum import env_paths
from nphm_tum.models.neural3dmm import construct_n3dmm, load_checkpoint

from nphm_tum.models.reconstruction import get_logits, get_vertex_color
from nphm_tum.utils.reconstruction import create_grid_points_from_bounds, mesh_from_logits


torch.cuda.manual_seed(0)
np.random.seed(0)
torch.random.seed()
torch.manual_seed(0)
random.seed(0)


def inverse_rendering(net,
                      subject,
                      expressions,
                      parameters,
                      num_views,
                      out_dir=None,
                      fine_smooth : bool = False,
                      intrinsics_provided : bool = True,
                      disable_mouth : bool = True,
                      lambda_normals : float = 0.0,
                      progress_interval : int = 300,
                      ):

    # output folders, one for each frame
    if fine_smooth:
        exp_dirs = [f'{out_dir}/{subject}/stage2/{ex:04d}/' for ex in expressions]
        exp_dir_stage1 = [f'{out_dir}/{subject}/stage1/{ex:04d}/' for ex in expressions]
        exp_dir_stage1_frame1 = f'{out_dir}/{subject}/stage1/{expressions[0]:04d}/'
    else:
        exp_dirs = [f'{out_dir}/{subject}/stage1/{ex:04d}/' for ex in expressions]

    if not fine_smooth:
        # prepare latent codes for optimization
        # TODO move this into tracker object!
        fix_id = False
        if parameters['lat_rep_id'] is not None:
            fix_id = True
            for k in parameters['lat_rep_id'].keys():
                parameters['lat_rep_id'][k] = torch.from_numpy(parameters['lat_rep_id'][k].detach().cpu().numpy()).cuda()
                parameters['lat_rep_id'][k].requires_grad = True
            latent_code = parameters['lat_rep_id']
            if type(parameters['lat_rep_expr']) is list:
                lat_rep_exp = parameters['lat_rep_expr'][0].squeeze()
            else:
                lat_rep_exp = parameters['lat_rep_expr'].squeeze()
        else:
            lat_rep_shape = torch.zeros([1, 1, net.id_model.lat_dim_glob + net.id_model.lat_dim_loc_geo * (net.id_model.n_anchors + 1)]).cuda()
            lat_rep_shape.requires_grad = True

            lat_rep_exp = torch.zeros([net.ex_model.lat_dim_expr]).cuda()
            lat_rep_exp.requires_grad = True
            lat_rep_app = torch.zeros([1, 1, net.id_model.lat_dim_glob + net.id_model.lat_dim_loc_app * (net.id_model.n_anchors + 1)]).cuda()
            lat_rep_app.requires_grad = True


            latent_code = {'geo': lat_rep_shape,
                            'app': lat_rep_app}
        expr_codebook = torch.nn.Embedding(num_embeddings=len(expressions), embedding_dim=lat_rep_exp.shape[0], sparse=True)
        expr_codebook.weight = torch.nn.Parameter(lat_rep_exp.unsqueeze(0).repeat(len(expressions), 1))

        wrapped_net = MonoNPHM(net, latent_code=latent_code, latent_codes_expr=expr_codebook)

        # if z_geo exists this frame was already processed from prev. frame
        # Load results and return it, s.t. next frames can be fitted
        if os.path.exists(exp_dirs[0] + 'z_geo.npy'):
            # affine color correction
            if os.path.exists(exp_dirs[0] + 'colorA.npy'):
                colorA = np.load(exp_dirs[0] + 'colorA.npy')
                colorb = np.load(exp_dirs[0] + 'colorb.npy')
            else:
                colorA, colorb = None, None
            # head pose correction w.r.t. to initial flame estimate
            if os.path.exists(exp_dirs[0] + 'scale.npy'):
                scale = np.load(exp_dirs[0] + 'scale.npy')
                rot = np.load(exp_dirs[0] + 'rot.npy')
                trans = np.load(exp_dirs[0] + 'trans.npy')
                params_pose = [rot, trans, scale]
                if len(expressions) > 1:
                    params_pose = [params_pose] + list([None, ] * (len(expressions) - 1))
                else:
                    params_pose = [params_pose]
            else:
                params_pose = None
            # Spherical Harmonics
            if os.path.exists(exp_dirs[0] + 'sh_coeffs.npy'):
                sh_coeffs = np.load(exp_dirs[0] + 'sh_coeffs.npy')
            else:
                sh_coeffs = None

            return {
                'lat_rep_id': {'geo': torch.from_numpy(np.load(exp_dirs[0] + 'z_geo.npy')).cuda(),
                               'app': torch.from_numpy(np.load(exp_dirs[0] + 'z_app.npy')).cuda()
                               },
                'lat_rep_expr': [torch.from_numpy(np.load(exp_dirs[0] + 'z_exp.npy')).cuda()],
                'colorA': colorA,
                'colorb': colorb,
                'params_pose': params_pose,
                'sh_coeffs': sh_coeffs,
            }


        # load hyperparameters for tracking
        cfg = f'{env_paths.CODE_BASE}/scripts/inference/configs/stage1.yaml'
        with open(cfg, 'r') as f:
            cfg = yaml.safe_load(f)

        # for tracking in the MonoNPHM paper we assume that there is only a weak facial expression for the first frame
        # We leverage this fact by setting a higher regularization weight to the expression
        # This will encourage the network to explain the observations by adapting the ID-code more strongly
        # For single frame fitting with strong expression one might need to remove/lower this weight
        if not fix_id:
            cfg['opt']['lambda_reg_expr'] = 1000

        tracker = TrackerMonoNPHM(
        net=wrapped_net,
        cfg=cfg,
        seq_name=subject,
        timesteps=expressions,
        num_views=num_views,
        exp_dirs=exp_dirs,
        fix_id =fix_id,
        colorA = parameters['colorA'],
        colorb = parameters['colorb'],
        params_pose = parameters['params_pose'],
        lr_scale = 0.99 if fix_id else 1,
        sh_coeffs = parameters['sh_coeffs'],
        stage2 = False,
        progress_interval = progress_interval,
        intrinsics_provided = intrinsics_provided,
            disable_mouth=disable_mouth,
            lambda_normals=lambda_normals
        )
        # perform tracking
        parameters = tracker.run_tracking()

    # TODO: ATTENTION: code for second stage tracking is not yet fully maintained in this repository
    else:
        cfg = f'{env_paths.CODE_BASE}/scripts/inference/configs/stage2.yaml'
        with open(cfg, 'r') as f:
            cfg = yaml.safe_load(f)

        rec_res = 250
        if rec_res is not None:
            cfg['reconstruction']['res'] = rec_res

        lat_rep_shape = torch.from_numpy(np.load(exp_dir_stage1_frame1 + 'z_geo.npy')).cuda()
        lat_rep_app = torch.from_numpy(np.load(exp_dir_stage1_frame1 + 'z_app.npy')).cuda()

        lat_reps_exp = []
        pose_params = []
        for e in expressions:
            lat_rep_exp = torch.from_numpy(np.load(exp_dir_stage1[e] + f'z_exp.npy')).cuda()
            lat_rep_exp.requires_grad = True
            lat_reps_exp.append(lat_rep_exp.squeeze())


            scale = np.load(exp_dir_stage1_frame1 + 'scale.npy')

            rot = np.load(exp_dir_stage1[e] + 'rot.npy')
            trans = np.load(exp_dir_stage1[e] + 'trans.npy')
            params_pose = [rot, trans, scale]
            pose_params.append(params_pose)
        expr_codebook = torch.nn.Embedding(num_embeddings=len(lat_reps_exp),
                                           embedding_dim=lat_reps_exp[0].shape[0],
                                           sparse=True)
        lat_reps_exp = torch.stack(lat_reps_exp, dim=0)
        expr_codebook.weight = torch.nn.Parameter(lat_reps_exp)


        latent_code = {'geo': lat_rep_shape,
                       'app': lat_rep_app}

        wrapped_net = MonoNPHM(net, latent_code=latent_code, latent_codes_expr=expr_codebook)

        colorA = np.load(exp_dir_stage1_frame1 + 'colorA.npy')
        colorb = np.load(exp_dir_stage1_frame1 + 'colorb.npy')
        if os.path.exists(exp_dir_stage1_frame1 + 'sh_coeffs.npy'):
            sh_coeffs = np.load(exp_dir_stage1_frame1 + 'sh_coeffs.npy')
        else:
            sh_coeffs = None

        parameters['colorA'] = colorA
        parameters['colorb'] = colorb
        parameters['sh_coeffs'] = sh_coeffs
        parameters['params_pose'] = pose_params

        tracker = TrackerMonoNPHM(
            net=wrapped_net,
            cfg=cfg,
            seq_name=subject,
            timesteps=expressions,
            num_views=num_views,
            exp_dirs=exp_dirs,
            fix_id=True, #fix_id,
            colorA=parameters['colorA'],
            colorb=parameters['colorb'],
            params_pose=parameters['params_pose'],
            lr_scale=0.99, # if fix_id else 1,
            sh_coeffs=parameters['sh_coeffs'],
            stage2=True,
            progress_interval=progress_interval,
            intrinsics_provided=intrinsics_provided,
            lambda_normals=lambda_normals
        )
        parameters = tracker.run_tracking()


    return parameters

def redo_mc(net,
                      subject,
                      expressions,
                      parameters,
                      num_views,
                      out_dir=None,
                      fine_smooth : bool = False,
                      ):
    reconstruction_cfg = {
    'min': [-.55, -.5, -.95],
    'max': [0.55, 0.75, 0.4],
    'res': 350  # small for faster reconstruction # use 256 or higher to grasp reconstructed geometry better
    }
    grid_points = create_grid_points_from_bounds(minimun=reconstruction_cfg['min'],
                                                 maximum=reconstruction_cfg['max'],
                                                 res=reconstruction_cfg['res'])
    grid_points = torch.from_numpy(grid_points).cuda().float()
    grid_points = torch.reshape(grid_points, (1, len(grid_points), 3)).cuda()
    write_dir = '/home/giebenhain/text_redo_mc/'
    os.makedirs(write_dir, exist_ok=True)
    for ex in expressions:
        # output folders, one for each frame
        exp_dir = f'{out_dir}/{subject}/stage2/{ex:04d}/'



        lat_rep_shape = torch.from_numpy(np.load(exp_dir+ 'z_geo.npy')).cuda()
        lat_rep_app = torch.from_numpy(np.load(exp_dir + 'z_app.npy')).cuda()

        lat_rep_exp = torch.from_numpy(np.load(exp_dir + f'z_exp.npy')).cuda()
        condition = {'geo': lat_rep_shape,
                     'app': lat_rep_app,
                     'exp': lat_rep_exp[ex].unsqueeze(0).unsqueeze(0)}

        colorA = torch.from_numpy(np.load(exp_dir + 'colorA.npy')).cuda()
        colorb = torch.from_numpy(np.load(exp_dir + 'colorb.npy')).cuda()



        logits = get_logits(net, condition, grid_points, nbatch_points=40000)
        mesh = mesh_from_logits(logits.copy(), reconstruction_cfg['min'], reconstruction_cfg['max'],reconstruction_cfg['res'])

        vertex_color = get_vertex_color(net,
                                        encoding=condition,
                                        vertices=torch.from_numpy(mesh.vertices).float().unsqueeze(0).cuda(),
                                        nbatch_points=40000,
                                        uniform_scaling=True,
                                        )
        vertex_color = ((vertex_color / 255) - 0.5) * 2
        if colorA is not None:
            vertex_color = vertex_color @ colorA.detach().cpu().numpy() + colorb.detach().cpu().numpy()
        vertex_color = ((vertex_color + 1) / 2 * 255).astype(np.uint8)
        mesh.visual.vertex_colors = vertex_color
        mesh.export(write_dir + f'{ex}.ply')
        torch.cuda.empty_cache()


    return parameters



def main(
        seq_tag: str, # name of sequence to track
        exp_name: str, # name of MonoNPHM model
        ckpt: int, # epoch from which to load checkpoint
        fine_smooth: bool = False, # toggle for first and second stage fitting
        rec_res : Optional[int] = None,
        progress_interval : int = 300,
        run_tag : Optional[str] = None,
        intrinsics_provided : bool = False,
        disable_mouth : bool = False,
        lambda_normals : float = 0.0,
):



    # load model config files
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
    modalities = ['geo', 'exp', 'app']
    n_lats = [len(subject_index), len(expression_index), len(subject_index)]

    neural_3dmm, latent_codes = construct_n3dmm(
        cfg=CFG,
        modalities=modalities,
        n_latents=n_lats,
        device=device,
        include_color_branch=True
    )

    # load checkpoint from trained NPHM model, including the latent codes
    ckpt_path = osp.join(weight_dir_shape, 'checkpoints/checkpoint_epoch_{}.tar'.format(ckpt))
    print('Loaded checkpoint from: {}'.format(ckpt_path))
    load_checkpoint(ckpt_path, neural_3dmm, latent_codes)

    # declare start and final frame of captured kinect sequences
    # "None" means that all available frames are used
    challeng_expressions = {
        'simon_507_s2': (0, None),
        'simon_507_s3': (0, None),
        'simon_507_s4': (0, None),
        'simon_507_s5': (0, None),
        'simon_508_s2': (0, None),
        'simon_508_s3': (0, None),
        'simon_508_s4': (0, None),
        'simon_508_s5': (0, None),
        'simon_509_s2': (0, None),
        'simon_509_s3': (16, None),
        'simon_509_s4': (0, None),
        'simon_509_s5': (0, None),
        'simon_510_s2': (0, 170),
        'simon_510_s3': (0, 180),
        'simon_510_s4': (0, None),
        'simon_510_s5': (0, None),
        'simon_511_s2': (0, None),
        'simon_511_s3': (0, None),
        'simon_511_s4': (0, None),
        'simon_511_s5': (0, None),
        'th_000': (0, None),
        'th_001': (0, None),
        'th_002': (0, None),
        'th_003': (0, None),
        'th_004': (0, None),
        'th_005': (0, None),
    }


    if seq_tag not in challeng_expressions:
        print('WARNING: no start and end of sequence specified. Usind full Sequence as default.')
        challeng_expressions[seq_tag] = (0, None)


    # create directory for output
    out_dir = f'{env_paths.TRACKING_RESULTS}/{exp_name}/'
    out_dir += f'{run_tag}'
    if lambda_normals > 0:
        out_dir += f'_lamNorm-{lambda_normals}'
    out_dir += '/'
    os.makedirs(out_dir, exist_ok=True)



    NUM_VIEWS = 1 # always just one view of the scene is used/available

    parameters = {
        'lat_rep_id': None,
        'lat_rep_expr': None,
        'colorA': None,
        'colorb': None,
        'params_pose': None,
        'sh_coeffs': None,
    }


    # check number of available frames
    #files = os.listdir(f'{env_paths.MICA_TRACKER_OUTPUT_PATH}/s{seq_tag[6:]}/video/')
    files = os.listdir(f'{env_paths.DATA_TRACKING}/{seq_tag}/metrical_tracker/{seq_tag}/video/')
    last_timestep = len(files)
    if challeng_expressions[seq_tag][1] is not None:
        last_timestep = challeng_expressions[seq_tag][1]
    expressions = range(challeng_expressions[seq_tag][0], last_timestep, 1)
    if fine_smooth:

        # wandb.init(project='inverse_rendering', tags=[f's{subject}_{seq_name}_fine'])

        parameters = inverse_rendering(neural_3dmm,
                                             seq_tag,
                                             expressions,
                                             parameters,
                                             num_views=NUM_VIEWS,
                                             out_dir=out_dir,
                                             fine_smooth=True,
                                             intrinsics_provided=intrinsics_provided,
                                             disable_mouth=disable_mouth,
                                             lambda_normals=lambda_normals,
                                             progress_interval=progress_interval,
                                             )  #
        # wandb.finish()
    else:
        # wandb.init(project='inverse_rendering', tags=[f's{subject}_{seq_name}'])

        for e, expression in enumerate(expressions):

            parameters = inverse_rendering(neural_3dmm,
                                             seq_tag,
                                             [expression],
                                             parameters,
                                             num_views=NUM_VIEWS,
                                             out_dir=out_dir,
                                             intrinsics_provided=intrinsics_provided,
                                             disable_mouth=disable_mouth,
                                             lambda_normals=lambda_normals,
                                             progress_interval=progress_interval,
                                             )
            # wandb.finish()
        # wandb.finish()

if __name__ == '__main__':
    tyro.cli(main)