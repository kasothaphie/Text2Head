# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2023 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de

import os.path
from enum import Enum
from functools import reduce
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from loguru import logger
from pytorch3d.io import load_obj
from pytorch3d.renderer import RasterizationSettings, PointLights, MeshRenderer, MeshRasterizer, TexturesVertex, SoftPhongShader, look_at_view_transform, PerspectiveCameras, BlendParams
from pytorch3d.structures import Meshes
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from pytorch3d.utils import opencv_from_cameras_projection
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import util
from configs.config import parse_args
from datasets.generate_dataset import GeneratorDataset
from datasets.image_dataset import ImagesDataset
from face_detector import FaceDetector
from flame.FLAME import FLAME, FLAMETex
from image import tensor2im
from renderer import Renderer

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
rank = 42
torch.manual_seed(rank)
torch.cuda.manual_seed(rank)
cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(rank)
I = torch.eye(3)[None].cuda().detach()
I6D = matrix_to_rotation_6d(I)
mediapipe_idx = np.load('flame/mediapipe/mediapipe_landmark_embedding.npz', allow_pickle=True, encoding='latin1')['landmark_indices'].astype(int)
left_iris_flame = [4597, 4542, 4510, 4603, 4570]
right_iris_flame = [4051, 3996, 3964, 3932, 4028]
left_iris_mp = [468, 469, 470, 471, 472]
right_iris_mp = [473, 474, 475, 476, 477]


class View(Enum):
    GROUND_TRUTH = 1
    COLOR_OVERLAY = 2
    SHAPE_OVERLAY = 4
    SHAPE = 8
    LANDMARKS = 16
    HEATMAP = 32
    DEPTH = 64


class Tracker(object):
    def __init__(self, config, device='cuda:0'):
        self.config = config
        self.device = device
        self.face_detector = FaceDetector('google')
        self.pyr_levels = config.pyr_levels
        self.cameras = PerspectiveCameras()#in_ndc=False)
        self.actor_name = self.config.config_name
        self.kernel_size = self.config.kernel_size
        self.sigma = None if self.config.sigma == -1 else self.config.sigma
        self.global_step = 0
        self.intrinsics_provided = self.config.intrinsics_provided



        logger.add(os.path.join(self.config.save_folder, self.actor_name, 'train.log'))

        # Latter will be set up
        self.frame = 0
        self.is_initializing = False
        self.image_size = torch.tensor([[config.image_size[0], config.image_size[1]]]).cuda()
        self.save_folder = self.config.save_folder
        self.output_folder = os.path.join(self.save_folder, self.actor_name)
        self.checkpoint_folder = os.path.join(self.save_folder, self.actor_name, "checkpoint")
        self.input_folder = os.path.join(self.save_folder, self.actor_name, "input")
        self.pyramid_folder = os.path.join(self.save_folder, self.actor_name, "pyramid")
        self.mesh_folder = os.path.join(self.save_folder, self.actor_name, "mesh")
        self.depth_folder = os.path.join(self.save_folder, self.actor_name, "depth")
        self.create_output_folders()
        self.writer = SummaryWriter(log_dir=self.save_folder + self.actor_name + '/logs')

        self.data_generator = GeneratorDataset(self.config.actor, self.config)
        self.prepare_data()

        self.config.image_size = (self.data_generator.config.image_size[0], self.data_generator.config.image_size[1])
        self.image_size = torch.tensor([[self.data_generator.config.image_size[0], self.data_generator.config.image_size[1]]], device=self.device)

        self.setup_renderer()

    def get_image_size(self):
        return self.image_size[0][0].item(), self.image_size[0][0].item()

    def create_output_folders(self):
        Path(self.save_folder).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_folder).mkdir(parents=True, exist_ok=True)
        Path(self.depth_folder).mkdir(parents=True, exist_ok=True)
        Path(self.mesh_folder).mkdir(parents=True, exist_ok=True)
        Path(self.input_folder).mkdir(parents=True, exist_ok=True)
        Path(self.pyramid_folder).mkdir(parents=True, exist_ok=True)

    def setup_renderer(self):
        mesh_file = 'data/head_template_mesh.obj'
        self.config.image_size = self.get_image_size()
        self.flame = FLAME(self.config).to(self.device)
        self.flametex = FLAMETex(self.config).to(self.device)
        self.diff_renderer = Renderer(self.image_size, obj_filename=mesh_file).to(self.device)
        self.faces = load_obj(mesh_file)[1]

        raster_settings = RasterizationSettings(
            image_size=self.get_image_size(),
            faces_per_pixel=1,
            cull_backfaces=True,
            perspective_correct=True
        )

        self.lights = PointLights(
            device=self.device,
            location=((0.0, 0.0, 5.0),),
            ambient_color=((0.5, 0.5, 0.5),),
            diffuse_color=((0.5, 0.5, 0.5),)
        )

        self.mesh_rasterizer = MeshRasterizer(raster_settings=raster_settings)
        self.debug_renderer = MeshRenderer(
            rasterizer=self.mesh_rasterizer,
            shader=SoftPhongShader(device=self.device, lights=self.lights)
        )

    def load_checkpoint(self, idx=-1):
        if not os.path.exists(self.checkpoint_folder):
            return False
        snaps = sorted(glob(self.checkpoint_folder + '/*.frame'))
        if len(snaps) == 0:
            logger.info('Training from beginning...')
            return False
        if len(snaps) == len(self.dataset):
            logger.info('Training has finished...')
            exit(0)

        last_snap = snaps[idx]
        payload = torch.load(last_snap)

        camera_params = payload['camera']
        self.R = nn.Parameter(torch.from_numpy(camera_params['R']).to(self.device))
        self.t = nn.Parameter(torch.from_numpy(camera_params['t']).to(self.device))
        if self.intrinsics_provided:
            self.focal_length = torch.from_numpy(camera_params['fl']).to(self.device)
            self.principal_point = torch.from_numpy(camera_params['pp']).to(self.device)
        else:
            self.focal_length = nn.Parameter(torch.from_numpy(camera_params['fl']).to(self.device))
            self.principal_point = nn.Parameter(torch.from_numpy(camera_params['pp']).to(self.device))

        flame_params = payload['flame']
        self.tex = nn.Parameter(torch.from_numpy(flame_params['tex']).to(self.device))
        self.exp = nn.Parameter(torch.from_numpy(flame_params['exp']).to(self.device))
        self.sh = nn.Parameter(torch.from_numpy(flame_params['sh']).to(self.device))
        self.shape = nn.Parameter(torch.from_numpy(flame_params['shape']).to(self.device))
        self.mica_shape = nn.Parameter(torch.from_numpy(flame_params['shape']).to(self.device))
        self.eyes = nn.Parameter(torch.from_numpy(flame_params['eyes']).to(self.device))
        self.eyelids = nn.Parameter(torch.from_numpy(flame_params['eyelids']).to(self.device))
        self.jaw = nn.Parameter(torch.from_numpy(flame_params['jaw']).to(self.device))

        self.frame = int(payload['frame_id'])
        self.global_step = payload['global_step']
        self.update_prev_frame()
        self.image_size = torch.from_numpy(payload['img_size'])[None].to(self.device)
        self.setup_renderer()

        logger.info(f'Snapshot loaded for frame {self.frame}')

        return True

    def save_checkpoint(self, frame_id):
        opencv = opencv_from_cameras_projection(self.cameras, self.image_size)

        frame = {
            'flame': {
                'exp': self.exp.clone().detach().cpu().numpy(),
                'shape': self.shape.clone().detach().cpu().numpy(),
                'tex': self.tex.clone().detach().cpu().numpy(),
                'sh': self.sh.clone().detach().cpu().numpy(),
                'eyes': self.eyes.clone().detach().cpu().numpy(),
                'eyelids': self.eyelids.clone().detach().cpu().numpy(),
                'jaw': self.jaw.clone().detach().cpu().numpy()
            },
            'camera': {
                'R': self.R.clone().detach().cpu().numpy(),
                't': self.t.clone().detach().cpu().numpy(),
                'fl': self.focal_length.clone().detach().cpu().numpy(),
                'pp': self.principal_point.clone().detach().cpu().numpy(),
            },
            'opencv': {
                'R': opencv[0].clone().detach().cpu().numpy(),
                't': opencv[1].clone().detach().cpu().numpy(),
                'K': opencv[2].clone().detach().cpu().numpy(),
            },
            'img_size': self.image_size.clone().detach().cpu().numpy()[0],
            'frame_id': frame_id,
            'global_step': self.global_step
        }

        vertices, _, _ = self.flame(
            cameras=torch.inverse(self.cameras.R),
            shape_params=self.shape,
            expression_params=self.exp,
            eye_pose_params=self.eyes,
            jaw_pose_params=self.jaw,
            eyelid_params=self.eyelids
        )

        f = self.diff_renderer.faces[0].cpu().numpy()
        v = vertices[0].cpu().numpy()

        trimesh.Trimesh(faces=f, vertices=v, process=False).export(f'{self.mesh_folder}/{frame_id}.ply')
        torch.save(frame, f'{self.checkpoint_folder}/{frame_id}.frame')
        np.savez(f'{self.checkpoint_folder}/{frame_id}_cam_params_opencv.npz', R=frame['opencv']['R'], t=frame['opencv']['t'], K=frame['opencv']['K'])

    def save_canonical(self):
        canon = os.path.join(self.save_folder, self.actor_name, "canonical.obj")
        if not os.path.exists(canon):
            from scipy.spatial.transform import Rotation as R
            rotvec = np.zeros(3)
            rotvec[0] = 12.0 * np.pi / 180.0
            jaw = matrix_to_rotation_6d(torch.from_numpy(R.from_rotvec(rotvec).as_matrix())[None, ...].cuda()).float()
            vertices = self.flame(cameras=torch.inverse(self.cameras.R), shape_params=self.shape, jaw_pose_params=jaw)[0].detach()
            faces = self.diff_renderer.faces[0].cpu().numpy()
            trimesh.Trimesh(faces=faces, vertices=vertices[0].cpu().numpy(), process=False).export(canon)

    def get_heatmap(self, values):
        l2 = tensor2im(values)
        l2 = cv2.cvtColor(l2, cv2.COLOR_RGB2BGR)
        l2 = cv2.normalize(l2, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(l2, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(cv2.addWeighted(heatmap, 0.75, l2, 0.25, 0).astype(np.uint8), cv2.COLOR_BGR2RGB) / 255.
        heatmap = torch.from_numpy(heatmap).permute(2, 0, 1)

        return heatmap

    def update_prev_frame(self):
        self.prev_R = self.R.clone().detach()
        self.prev_t = self.t.clone().detach()
        self.prev_exp = self.exp.clone().detach()
        self.prev_eyes = self.eyes.clone().detach()
        self.prev_jaw = self.jaw.clone().detach()

    def render_shape(self, vertices, faces=None, white=True):
        B = vertices.shape[0]
        V = vertices.shape[1]
        if faces is None:
            faces = self.faces.verts_idx.cuda()[None].repeat(B, 1, 1)
        if not white:
            verts_rgb = torch.from_numpy(np.array([80, 140, 200]) / 255.).cuda().float()[None, None, :].repeat(B, V, 1)
        else:
            verts_rgb = torch.from_numpy(np.array([1.0, 1.0, 1.0])).cuda().float()[None, None, :].repeat(B, V, 1)
        textures = TexturesVertex(verts_features=verts_rgb.cuda())
        meshes_world = Meshes(verts=[vertices[i] for i in range(B)], faces=[faces[i] for i in range(B)], textures=textures)

        blend = BlendParams(background_color=(1.0, 1.0, 1.0))

        fragments = self.mesh_rasterizer(meshes_world, cameras=self.cameras)
        rendering = self.debug_renderer.shader(fragments, meshes_world, cameras=self.cameras, blend_params=blend)
        rendering = rendering.permute(0, 3, 1, 2).detach()
        return rendering[:, 0:3, :, :]

    def to_cuda(self, batch, unsqueeze=False):
        for key in batch.keys():
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(self.device)
                if unsqueeze:
                    batch[key] = batch[key][None]

        return batch

    def create_parameters(self):
        bz = 1
        R, T = look_at_view_transform(dist=1.0)
        self.R = nn.Parameter(matrix_to_rotation_6d(R).to(self.device))
        self.t = nn.Parameter(T.to(self.device))
        self.shape = nn.Parameter(self.mica_shape)
        self.mica_shape = nn.Parameter(self.mica_shape)
        self.tex = nn.Parameter(torch.zeros(bz, self.config.tex_params).float().to(self.device))
        self.exp = nn.Parameter(torch.zeros(bz, self.config.num_exp_params).float().to(self.device))
        self.sh = nn.Parameter(torch.zeros(bz, 9, 3).float().to(self.device))
        #TODO: provided camera intrinsics
        if self.intrinsics_provided: # Kinect intrinsics in ndc format
            self.focal_length = torch.tensor([[457.05*2/1080*512/512*2]]).float().to(self.device) #nn.Parameter(torch.tensor([[5000 / self.get_image_size()[0]]]).to(self.device))
            self.principal_point = torch.tensor([[269.5125*2/1080*512/512*2-1, 270.869*2/1080*512/512*2-1]]).float().to(self.device)
        else:
            self.focal_length = nn.Parameter(torch.tensor([[5000 / self.get_image_size()[0]]]).to(self.device))
            self.principal_point = nn.Parameter(torch.zeros(bz, 2).float().to(self.device))
        self.eyes = nn.Parameter(torch.cat([matrix_to_rotation_6d(I), matrix_to_rotation_6d(I)], dim=1))
        self.jaw = nn.Parameter(matrix_to_rotation_6d(I))
        self.eyelids = nn.Parameter(torch.zeros(bz, 2).float().to(self.device))

    @staticmethod
    def save_tensor(tensor, path='tensor.jpg'):
        img = (tensor[0].detach().cpu().numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
        img = np.minimum(np.maximum(img, 0), 255).astype(np.uint8)
        cv2.imwrite(path, img)

    def parse_mask(self, ops, batch, visualization=False):
        _, _, h, w = ops['alpha_images'].shape
        result = ops['mask_images_rendering']

        if visualization:
            result = ops['mask_images']

        return result.detach()

    def update(self, param_groups):
        for param in param_groups:
            for i, name in enumerate(param['name']):
                setattr(self, name, nn.Parameter(param['params'][i].clone().detach()))

    def get_param(self, name, param_groups):
        for param in param_groups:
            if name in param['name']:
                return param['params'][param['name'].index(name)]
        return getattr(self, name)

    def clone_params_tracking(self):
        params = [
            {'params': [nn.Parameter(self.exp.clone())], 'lr': 0.025, 'name': ['exp']},
            {'params': [nn.Parameter(self.eyes.clone())], 'lr': 0.001, 'name': ['eyes']},
            {'params': [nn.Parameter(self.eyelids.clone())], 'lr': 0.001, 'name': ['eyelids']},
            {'params': [nn.Parameter(self.R.clone())], 'lr': self.config.rotation_lr, 'name': ['R']},
            {'params': [nn.Parameter(self.t.clone())], 'lr': self.config.translation_lr, 'name': ['t']},
            {'params': [nn.Parameter(self.sh.clone())], 'lr': 0.001, 'name': ['sh']}
        ]

        if self.config.optimize_jaw:
            params.append({'params': [nn.Parameter(self.jaw.clone().detach())], 'lr': 0.001, 'name': ['jaw']})

        return params

    def clone_params_initialization(self):
        params = [
            {'params': [nn.Parameter(self.exp.clone())], 'lr': 0.025, 'name': ['exp']},
            {'params': [nn.Parameter(self.eyes.clone())], 'lr': 0.001, 'name': ['eyes']},
            {'params': [nn.Parameter(self.eyelids.clone())], 'lr': 0.01, 'name': ['eyelids']},
            {'params': [nn.Parameter(self.sh.clone())], 'lr': 0.01, 'name': ['sh']},
            {'params': [nn.Parameter(self.t.clone())], 'lr': 0.005, 'name': ['t']},
            {'params': [nn.Parameter(self.R.clone())], 'lr': 0.005, 'name': ['R']},

        ]
        if not self.intrinsics_provided:
            params.append({'params': [nn.Parameter(self.principal_point.clone())], 'lr': 0.001, 'name': ['principal_point']})
            params.append({'params': [nn.Parameter(self.focal_length.clone())], 'lr': 0.001, 'name': ['focal_length']})

        if self.config.optimize_shape:
            params.append({'params': [nn.Parameter(self.shape.clone().detach())], 'lr': 0.025, 'name': ['shape']})

        if self.config.optimize_jaw:
            params.append({'params': [nn.Parameter(self.jaw.clone().detach())], 'lr': 0.001, 'name': ['jaw']})

        return params

    def clone_params_color(self):
        params = [
            {'params': [nn.Parameter(self.sh.clone())], 'lr': 0.05, 'name': ['sh']},
            {'params': [nn.Parameter(self.tex.clone())], 'lr': 0.05, 'name': ['tex']},
        ]

        return params

    @staticmethod
    def reduce_loss(losses):
        all_loss = 0.
        for key in losses.keys():
            all_loss = all_loss + losses[key]
        losses['all_loss'] = all_loss
        return all_loss

    def optimize_camera(self, batch, steps=1000):
        batch = self.to_cuda(batch)
        images, landmarks, landmarks_dense, lmk_dense_mask, lmk_mask = self.parse_batch(batch)

        h, w = images.shape[2:4]
        self.shape = batch['shape']
        self.mica_shape = batch['shape'].clone().detach()  # Save it for regularization

        # Important to initialize
        self.create_parameters()

        if not self.intrinsics_provided:
            params = [{'params': [self.t, self.R, self.focal_length, self.principal_point], 'lr': 0.05}]
        else:
            params = [{'params': [self.t, self.R], 'lr': 0.05}]

        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)

        t = tqdm(range(steps), desc='', leave=True, miniters=100)
        for k in t:
            self.cameras = PerspectiveCameras(
                device=self.device,
                principal_point=self.principal_point,
                focal_length=self.focal_length,
                R=rotation_6d_to_matrix(self.R), T=self.t,
                image_size=self.image_size,
                #in_ndc=False
            )
            _, lmk68, lmkMP = self.flame(cameras=torch.inverse(self.cameras.R), shape_params=self.shape, expression_params=self.exp, eye_pose_params=self.eyes, jaw_pose_params=self.jaw)
            points68 = self.cameras.transform_points_screen(lmk68)[..., :2]
            pointsMP = self.cameras.transform_points_screen(lmkMP)[..., :2]

            losses = {}
            losses['pp_reg'] = torch.sum(self.principal_point ** 2)
            losses['lmk68'] = util.lmk_loss(points68, landmarks[..., :2], [h, w], lmk_mask) * self.config.w_lmks
            losses['lmkMP'] = util.lmk_loss(pointsMP, landmarks_dense[..., :2], [h, w], lmk_dense_mask) * self.config.w_lmks

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            scheduler.step()

            loss = all_loss.item()
            # self.writer.add_scalar('camera', loss, global_step=k)
            t.set_description(f'Loss for camera {loss:.4f}')
            self.frame += 1
            if k % 100 == 0 and k > 0:
                self.checkpoint(batch, visualizations=[[View.GROUND_TRUTH, View.LANDMARKS, View.SHAPE_OVERLAY]], frame_dst='/camera', save=False, dump_directly=True)

        self.frame = 0

    def optimize_color(self, batch, pyramid, params_func, pho_weight_func, reg_from_prev=False):
        self.update_prev_frame()
        images, landmarks, landmarks_dense, lmk_dense_mask, lmk_mask = self.parse_batch(batch)

        aspect_ratio = util.get_aspect_ratio(images)
        h, w = images.shape[2:4]
        logs = []

        for k, level in enumerate(pyramid):
            img, iters, size, image_size = level
            # Optimizer per step
            optimizer = torch.optim.Adam(params_func())
            params = optimizer.param_groups

            shape = self.get_param('shape', params)
            exp = self.get_param('exp', params)
            eyes = self.get_param('eyes', params)
            eyelids = self.get_param('eyelids', params)
            jaw = self.get_param('jaw', params)
            tex = self.get_param('tex', params)
            sh = self.get_param('sh', params)
            t = self.get_param('t', params)
            R = self.get_param('R', params)
            fl = self.focal_length
            pp = self.principal_point

            scale = image_size[0] / h
            self.diff_renderer.set_size(size)
            self.debug_renderer.rasterizer.raster_settings.image_size = size
            flipped = torch.flip(img, [2, 3])

            image_lmks68 = landmarks * scale
            image_lmksMP = landmarks_dense * scale
            left_iris = batch['left_iris'] * scale
            right_iris = batch['right_iris'] * scale
            mask_left_iris = batch['mask_left_iris'] * scale
            mask_right_iris = batch['mask_right_iris'] * scale

            self.diff_renderer.rasterizer.reset()

            best_loss = np.inf

            for p in range(iters):
                if p % self.config.raster_update == 0:
                    self.diff_renderer.rasterizer.reset()
                losses = {}
                self.cameras = PerspectiveCameras(
                    device=self.device,
                    principal_point=pp,
                    focal_length=fl,
                    R=rotation_6d_to_matrix(R), T=t,
                    image_size=(image_size,),
                    #in_ndc=False
                )
                vertices, lmk68, lmkMP = self.flame(
                    cameras=torch.inverse(self.cameras.R),
                    shape_params=shape,
                    expression_params=exp,
                    eye_pose_params=eyes,
                    jaw_pose_params=jaw,
                    eyelid_params=eyelids
                )

                proj_lmksMP = self.cameras.transform_points_screen(lmkMP)[..., :2]
                proj_lmks68 = self.cameras.transform_points_screen(lmk68)[..., :2]
                proj_vertices = self.cameras.transform_points_screen(vertices)[..., :2]

                right_eye, left_eye = eyes[:, :6], eyes[:, 6:]

                # Landmarks sparse term
                losses['loss/lmk_oval'] = util.oval_lmk_loss(proj_lmks68, image_lmks68, image_size, lmk_mask) * self.config.w_lmks_oval
                losses['loss/lmk_68'] = util.lmk_loss(proj_lmks68, image_lmks68, image_size, lmk_mask) * self.config.w_lmks_68
                losses['loss/lmk_MP'] = util.face_lmk_loss(proj_lmksMP, image_lmksMP, image_size, True, lmk_dense_mask) * self.config.w_lmks
                losses['loss/lmk_eye'] = util.eye_closure_lmk_loss(proj_lmksMP, image_lmksMP, image_size, lmk_dense_mask) * self.config.w_lmks_lid
                losses['loss/lmk_mouth'] = util.mouth_lmk_loss(proj_lmksMP, image_lmksMP, image_size, True, lmk_dense_mask) * self.config.w_lmks_mouth
                losses['loss/lmk_iris_left'] = util.lmk_loss(proj_vertices[:, left_iris_flame, ...], left_iris, image_size, mask_left_iris) * self.config.w_lmks_iris
                losses['loss/lmk_iris_right'] = util.lmk_loss(proj_vertices[:, right_iris_flame, ...], right_iris, image_size, mask_right_iris) * self.config.w_lmks_iris

                # Reguralizers
                losses['reg/exp'] = torch.sum(exp ** 2) * self.config.w_exp
                losses['reg/sym'] = torch.sum((right_eye - left_eye) ** 2) * 8.0
                losses['reg/jaw'] = torch.sum((I6D - jaw) ** 2) * self.config.w_jaw
                losses['reg/eye_lids'] = torch.sum((eyelids[:, 0] - eyelids[:, 1]) ** 2)
                losses['reg/eye_left'] = torch.sum((I6D - left_eye) ** 2)
                losses['reg/eye_right'] = torch.sum((I6D - right_eye) ** 2)
                losses['reg/shape'] = torch.sum((shape - self.mica_shape) ** 2) * self.config.w_shape
                losses['reg/tex'] = torch.sum(tex ** 2) * self.config.w_tex
                losses['reg/pp'] = torch.sum(pp ** 2)

                # Dense term (look at the config pyr_levels)
                if k > 0 or self.is_initializing:
                    albedos = self.flametex(tex)
                    ops = self.diff_renderer(vertices, albedos, sh, self.cameras)

                    # Photometric dense term
                    grid = ops['position_images'].permute(0, 2, 3, 1)[:, :, :, :2]
                    sampled_image = F.grid_sample(flipped, grid * aspect_ratio, align_corners=False)

                    losses['loss/pho'] = util.pixel_loss(ops['images'], sampled_image, self.parse_mask(ops, batch)) * pho_weight_func(k)

                all_loss = self.reduce_loss(losses)
                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

                for key in losses.keys():
                    self.writer.add_scalar(key, losses[key], global_step=self.global_step)

                self.global_step += 1

                if p % iters == 0:
                    logs.append(f"Color loss for level {k} [frame {str(self.frame).zfill(4)}] =" + reduce(lambda a, b: a + f' {b}={round(losses[b].item(), 4)}', [""] + list(losses.keys())))

                loss_color = all_loss.item()

                if loss_color < best_loss:
                    best_loss = loss_color
                    self.update(optimizer.param_groups)

        for log in logs: logger.info(log)

    def checkpoint(self, batch, visualizations=[[View.GROUND_TRUTH, View.LANDMARKS, View.HEATMAP], [View.COLOR_OVERLAY, View.SHAPE_OVERLAY, View.SHAPE]], frame_dst='/video', save=True, dump_directly=False):
        batch = self.to_cuda(batch)
        images, landmarks, landmarks_dense, _, _ = self.parse_batch(batch)

        input_image = util.to_image(batch['image'].clone()[0].cpu().numpy())

        savefolder = self.save_folder + self.actor_name + frame_dst
        Path(savefolder).mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            self.cameras = PerspectiveCameras(
                device=self.device,
                principal_point=self.principal_point,
                focal_length=self.focal_length,
                R=rotation_6d_to_matrix(self.R), T=self.t,
                image_size=self.image_size,
                #in_ndc=False,
            )

            self.diff_renderer.rasterizer.reset()
            self.diff_renderer.set_size(self.get_image_size())
            self.debug_renderer.rasterizer.raster_settings.image_size = self.get_image_size()

            vertices, lmk68, lmkMP = self.flame(
                cameras=torch.inverse(self.cameras.R),
                shape_params=self.shape,
                expression_params=self.exp,
                eye_pose_params=self.eyes,
                jaw_pose_params=self.jaw,
                eyelid_params=self.eyelids
            )

            lmk68 = self.cameras.transform_points_screen(lmk68, image_size=self.image_size)
            lmkMP = self.cameras.transform_points_screen(lmkMP, image_size=self.image_size)

            albedos = self.flametex(self.tex)
            albedos = F.interpolate(albedos, self.get_image_size(), mode='bilinear')
            ops = self.diff_renderer(vertices, albedos, self.sh, cameras=self.cameras)
            mask = (self.parse_mask(ops, batch, visualization=True) > 0).float()
            predicted_images = (ops['images'] * mask + (images * (1.0 - mask)))[0]
            shape_mask = ((ops['alpha_images'] * ops['mask_images_mesh']) > 0.).int()[0]

            final_views = []

            for views in visualizations:
                row = []
                for view in views:
                    if True: #view == View.COLOR_OVERLAY:
                        row.append(ops['images'][0].cpu().numpy())
                    if view == View.GROUND_TRUTH:
                        row.append(images[0].cpu().numpy())
                    if view == View.SHAPE:
                        shape = self.render_shape(vertices, white=False)[0].cpu().numpy()
                        row.append(shape)
                    if view == View.LANDMARKS:
                        gt_lmks = images.clone()
                        gt_lmks = util.tensor_vis_landmarks(gt_lmks, torch.cat([landmarks_dense, landmarks[:, :17, :]], dim=1), color='g')
                        gt_lmks = util.tensor_vis_landmarks(gt_lmks, torch.cat([lmkMP, lmk68[:, :17, :]], dim=1), color='r')
                        row.append(gt_lmks[0].cpu().numpy())
                    if view == View.SHAPE_OVERLAY:
                        shape = self.render_shape(vertices, white=False)[0] * shape_mask
                        blend = images[0] * (1 - shape_mask) + images[0] * shape_mask * 0.3 + shape * 0.7 * shape_mask
                        row.append(blend.cpu().numpy())
                    #if True:
                    #    #blend = torch.zeros_like(images[0]) * (1 - shape_mask) + torch.ones_like(images[0]) * shape_mask #* 0.3 + shape * 0.7 * shape_mask
                    #    row.append(shape_mask.cpu().numpy())
                    #if True: #view == View.HEATMAP:
                    #    t = images[0].cpu()
                    #    f = predicted_images.cpu()
                    #    l2 = torch.pow(torch.abs(f - t), 2)
                    #    heatmap = self.get_heatmap(l2[None])
                    #    row.append(heatmap)
                final_views.append(row)

            # VIDEO
            final_views = util.merge_views(final_views)
            frame_id = str(self.frame).zfill(5)

            cv2.imwrite('{}/{}.jpg'.format(savefolder, frame_id), final_views)
            cv2.imwrite('{}/{}.png'.format(self.input_folder, frame_id), input_image)

            if not save:
                return

            # CHECKPOINT
            self.save_checkpoint(frame_id)

            # DEPTH
            depth_view = self.diff_renderer.render_depth(vertices, cameras=self.cameras, faces=torch.cat([util.get_flame_extra_faces(), self.diff_renderer.faces], dim=1))
            depth = depth_view[0].permute(1, 2, 0)[..., 2:].cpu().numpy() * 1000.0
            cv2.imwrite('{}/{}.png'.format(self.depth_folder, frame_id), depth.astype(np.uint16))

    def optimize_frame(self, batch):
        batch = self.to_cuda(batch)
        images = self.parse_batch(batch)[0]
        h, w = images.shape[2:4]
        pyramid_size = np.array([h, w])
        pyramid = util.get_gaussian_pyramid([(pyramid_size * size, util.round_up_to_odd(steps)) for size, steps in self.pyr_levels], images, self.kernel_size, self.sigma)
        self.optimize_color(batch, pyramid, self.clone_params_tracking, lambda k: self.config.w_pho, reg_from_prev=True)
        self.checkpoint(batch, visualizations=[[View.GROUND_TRUTH, View.COLOR_OVERLAY, View.LANDMARKS, View.SHAPE]])

    def optimize_video(self):
        self.is_initializing = False
        for i in list(range(self.frame, len(self.dataset))):
            batch = self.to_cuda(self.dataset[i], unsqueeze=True)
            if type(batch) is torch.Tensor:
                continue
            self.optimize_frame(batch)
            self.frame += 1

    def output_video(self):
        util.images_to_video(self.output_folder, self.config.fps)

    def parse_batch(self, batch):
        images = batch['image']
        landmarks = batch['lmk']
        landmarks_dense = batch['dense_lmk']

        lmk_dense_mask = ~(landmarks_dense.sum(2, keepdim=True) == 0)
        lmk_mask = ~(landmarks.sum(2, keepdim=True) == 0)

        left_iris = landmarks_dense[:, left_iris_mp, :]
        right_iris = landmarks_dense[:, right_iris_mp, :]
        mask_left_iris = lmk_dense_mask[:, left_iris_mp, :]
        mask_right_iris = lmk_dense_mask[:, right_iris_mp, :]

        batch['left_iris'] = left_iris
        batch['right_iris'] = right_iris
        batch['mask_left_iris'] = mask_left_iris
        batch['mask_right_iris'] = mask_right_iris

        return images, landmarks, landmarks_dense[:, mediapipe_idx, :2], lmk_dense_mask[:, mediapipe_idx, :], lmk_mask

    def prepare_data(self):
        self.data_generator.run()
        self.dataset = ImagesDataset(self.config)
        self.dataloader = DataLoader(self.dataset, batch_size=1, num_workers=0, shuffle=False, pin_memory=True, drop_last=False)

    def initialize_tracking(self):
        self.is_initializing = True
        keyframes = self.config.keyframes
        if len(keyframes) == 0:
            logger.error('[ERROR] Keyframes are empty!')
            exit(0)
        keyframes.insert(0, keyframes[0])
        for i, j in enumerate(keyframes):
            batch = self.to_cuda(self.dataset[j], unsqueeze=True)
            images = self.parse_batch(batch)[0]
            h, w = images.shape[2:4]
            pyramid_size = np.array([h, w])
            pyramid = util.get_gaussian_pyramid([(pyramid_size * size, util.round_up_to_odd(steps * 2)) for size, steps in self.pyr_levels], images, self.kernel_size, self.sigma)
            params = self.clone_params_initialization
            if i == 0:
                params = self.clone_params_color
                self.optimize_camera(batch)
                for k, level in enumerate(pyramid):
                    self.save_tensor(level[0], f"{self.pyramid_folder}/{k}.png")
            self.optimize_color(batch, pyramid, params, lambda k: self.config.w_pho)
            self.checkpoint(batch, visualizations=[[View.GROUND_TRUTH, View.COLOR_OVERLAY, View.LANDMARKS, View.SHAPE]], frame_dst='/initialization')
            self.frame += 1

        self.save_canonical()

    def run(self):
        if not self.load_checkpoint():
            self.initialize_tracking()
            self.frame = 0

        self.optimize_video()
        self.output_video()


if __name__ == '__main__':
    config = parse_args()
    ff = Tracker(config, device='cuda:0')
    ff.run()
