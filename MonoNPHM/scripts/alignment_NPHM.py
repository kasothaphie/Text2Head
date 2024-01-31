import numpy as np
import trimesh
import pymeshlab
import pyvista as pv
from PIL import Image
import tyro

from famudy.data import FamudySequenceDataManager

from nphm_tum import env_paths
from nphm_tum.utils.tranformations import invert_similarity_transformation, apply_transform
from nphm_tum.utils.alignment import get_transform_mvs2flame2nphm, load_camera_params
from nphm_tum.render_utils.pyvista_renderer import render



def check_alignment(p_id : int,
                    seq_name : str,
                    frame : int):
    mesh_dir = f'~/test_nphm_tum_repo/{p_id}_{seq_name}'
    flame_fitting_dir = env_paths.NERSEMBLE_DATASET_PATH + '/{:03d}/sequences/{}/annotations/tracking/'.format(p_id, seq_name)

    # load trimesh mesh, slightly more effort since it is in the CTM format
    mesh_path = f'{mesh_dir}/{frame:05d}.CTM'
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)
    pred_mesh = trimesh.Trimesh(ms.mesh(0).vertex_matrix(), ms.mesh(0).face_matrix(), process=False)

    # load transformation from NPHM coordinates into the coordinate system of the multi-view camera system
    # lazily try FLAME2023_v2 (most up to date) FLAME fitting and if it fails fall back to FLAME2023
    try:
        T_nphm2mvs = get_transform_mvs2flame2nphm(p_id, seq_name, flame_fitting_dir + '/FLAME2023_v2//tracked_flame_params.npz',
                                                  flame_fitting_dir + '/FLAME2023_v2_2_NPHM_corrective/corrective_transform.npz',
                                                  frame,
                                                  nphm_tracking_dir=flame_fitting_dir,
                                                  inverse=True)
    except Exception as ex:
        T_nphm2mvs = get_transform_mvs2flame2nphm(p_id, seq_name, flame_params_path=flame_fitting_dir + '/FLAME2023//tracked_flame_params.npz',
                                                  corrective_transform_path=flame_fitting_dir + '/FLAME2023_2_NPHM_corrective/corrective_transform.npz',
                                                  nphm_tracking_dir=flame_fitting_dir,
                                                  frame=frame,
                                                  inverse=True)
    pred_mesh = apply_transform(pred_mesh, T_nphm2mvs)


    # create the data manager
    manager = FamudySequenceDataManager(participant_id=p_id,
                                        sequence_name=seq_name,
                                        )

    # example: load and show image
    gt_image = manager.load_image(timestep=frame, cam_id_or_serial='222200037')

    I = Image.fromarray(gt_image)
    I.show()

    # load point cloud
    pc, colors, normals = manager.load_point_cloud(timestep=frame, n_cameras=16)

    # create and view 3D plotter that shows alignment of COLMAP point cloud and NPHM mesh
    pl = pv.Plotter()
    pl.add_points(pc)
    pl.add_mesh(pred_mesh)
    pl.show()

    # example: load all available timesteps
    all_timesteps = manager.get_timesteps()

    # example load camera parameters
    intrinsics, c2w = load_camera_params(manager)

    # example: render mesh from the perspective of camera "222200037" using pyvista as simple renderer
    pl = None
    image = render(pl, pred_mesh, c2w['222200037'], intrinsics['222200037'])
    I = Image.fromarray(image)
    I.show()

    # example: simple image overlay of rendered image and gt image
    composed = gt_image.copy()
    alpha_mask = image[..., 2] != 255
    composed[alpha_mask, :] = 0.6 * image[alpha_mask, :3] + 0.4 * composed[alpha_mask, :]
    I = Image.fromarray(composed)
    I.show()

if __name__ == '__main__':
    tyro.cli(check_alignment)