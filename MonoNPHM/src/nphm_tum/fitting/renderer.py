import torch
import torch.nn as nn
import pyvista as pv
import numpy as np
import trimesh
from pytorch3d.transforms import so3_exp_map


def get_sphere_intersection(cam_loc, ray_directions, r = 1.0):
    n_imgs, n_pix, _ = ray_directions.shape

    cam_loc = cam_loc.unsqueeze(-1)
    ray_cam_dot = torch.bmm(ray_directions, cam_loc).squeeze()
    under_sqrt = ray_cam_dot ** 2 - (cam_loc.norm(2,1) ** 2 - r ** 2)

    under_sqrt = under_sqrt.reshape(-1)
    mask_intersect = under_sqrt > 0
    sphere_intersections = torch.zeros(n_imgs * n_pix, 2).cuda().float()
    sphere_intersections[mask_intersect] = torch.sqrt(under_sqrt[mask_intersect]).unsqueeze(-1) * torch.Tensor([-1, 1]).cuda().float()
    sphere_intersections[mask_intersect] -= ray_cam_dot.reshape(-1)[mask_intersect].unsqueeze(-1)
    sphere_intersections[~mask_intersect] = torch.mean(sphere_intersections[mask_intersect], dim=0)

    sphere_intersections = sphere_intersections.reshape(n_imgs, n_pix, 2)
    sphere_intersections = sphere_intersections.clamp_min(0.0)
    mask_intersect = mask_intersect.reshape(n_imgs, n_pix)
    mask_intersect = torch.ones_like(mask_intersect)

    return sphere_intersections, mask_intersect


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-8  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples, device='cuda')
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device='cuda')
        u = torch.sort(u, dim=-1)[0]

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-8, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class RendererMonoNPHM(nn.Module):
    def __init__(self, implicit_network_forward, conf, sh_coeffs=None):
        super().__init__()
        self.implicit_network = implicit_network_forward
        self.object_bounding_sphere = conf['object_bounding_sphere']
        self.n_steps = conf['n_steps']

        # constant factor needed for SH computation
        pi = np.pi
        constant_factor = torch.tensor(
            [1 / np.sqrt(4 * pi), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), \
             ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))), (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi)))]).float()
        self.register_buffer('sh_constant_factor', constant_factor)

        # register sh parameters
        # if None initi as uniform lighting
        if sh_coeffs is None:
            sh_coeffs = torch.zeros(1, 9, 3).float().cuda()
            sh_coeffs[:, 0, :] = np.sqrt(4 * np.pi)
        else:
            sh_coeffs = torch.from_numpy(sh_coeffs).cuda()
        sh_coeffs = nn.Parameter(sh_coeffs)
        self.register_parameter('sh_coeffs', sh_coeffs)


    def add_SHlight(self, normal_images, sh_coeff):
        '''
            Compute SH shading

            normals: [nrays, nsamples, 3]
            sh_coeff: [1, 9, 3]
            self.constant_factor: [9]
        '''
        N = normal_images
        sh = torch.stack([
            N[:, :, 0] * 0. + 1., N[:, :, 0], N[:, :, 1],
            N[:, :, 2], N[:, :, 0] * N[:, :, 1], N[:, :, 0] * N[:, :, 2],
            N[:, :, 1] * N[:, :, 2], N[:, :, 0] ** 2 - N[:, :, 1] ** 2, 3 * (N[:, :, 2] ** 2) - 1
        ],
            2)  # [nrays, nsamples, 9]
        sh = sh * self.sh_constant_factor[None, None, :]
        shading = torch.sum(sh_coeff[None, :, :, :] * sh[:, :, :, None], 2)  # [bz, 9, 3,] before sum
        return shading # [bz, 3]


    def eval_sdf_inside_sphere(self,
                               num_pixels,
                               sdf,
                               cam_loc,
                               ray_directions,
                               mask,
                               min_dis,
                               max_dis,
                               samples=None,
                               pose_params=None,
                               chunk_size=25000):
        '''
        Evaluate sdf for uniformly sampled points within the unit sphere.
        Tensors min_dis and max_dis which specify the minimum and maximum distance where a ray enters/exists
        the bounding sphere.
        '''

        n_mask_points = mask.sum()
        n = self.n_steps

        # compute 'self.n_steps' points uniformly spaced between entering and exiting of bounding sphere
        # steps describes distance along a ray
        steps = torch.linspace(0.0, 1.0, n).cuda()
        mask_max_dis = max_dis[mask].unsqueeze(-1)
        mask_min_dis = min_dis[mask].unsqueeze(-1)
        steps = steps.unsqueeze(0).repeat(n_mask_points, 1) * (mask_max_dis - mask_min_dis) + mask_min_dis

        # add random term to samples
        sample_dist = (mask_max_dis - mask_min_dis) / n
        t_rand = (torch.rand(steps.shape, device=steps.device) - 0.5)
        steps = steps + t_rand * sample_dist


        # compute world space position of points that correspond to steps
        reshaped = False
        if len(mask.shape) > 1:
            mask = mask.squeeze()
            reshaped = True
        mask_points = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[mask]
        if reshaped:
            mask = mask.unsqueeze(0)
        mask_rays = ray_directions[mask, :]

        mask_points_all = mask_points.unsqueeze(1).repeat(1, n, 1) + steps.unsqueeze(-1) * mask_rays.unsqueeze(
            1).repeat(1, n, 1)
        points = mask_points_all.reshape(-1, 3)


        # eval network for all points in a chunked fashion
        mask_net_out_all = []
        for pnts in torch.split(points, chunk_size, dim=0): # increaseing chunk size can improve perofmance if enough GPU memory is available
            # if rigid head pose is present transfrom from wolrd space into canonical space
            if pose_params is not None:
                pnts = pose_params[2] * pnts @ so3_exp_map(pose_params[0]).squeeze().T + pose_params[1]
            if len(pnts.shape) == 4:
                pnts = pnts.squeeze(0)
            mask_net_out_all.append(sdf(pnts))

        mask_net_out_all = torch.cat(mask_net_out_all).reshape(-1, n, mask_net_out_all[0].shape[-1])
        return mask_net_out_all, steps.reshape(-1, n)


    def forward(self,
                input,
                expression,
                compute_non_convergent=False,
                skip_render=False,
                neus_variance=None,
                debug_plot=False,
                pose_params=None,
                use_SH = True,
                num_samples=32,
                vari=0.3,
                ):
        ray_dirs = input['ray_dirs']
        cam_loc = input['cam_loc']
        object_mask = input["object_mask"].reshape(-1)
        batch_size, num_pixels, _ = ray_dirs.shape
        w2c = input['w2c']

        if not skip_render:

            with torch.no_grad():
                # obtain intersections with bounding sphere
                sphere_intersections, mask_intersect = get_sphere_intersection(cam_loc,
                                                                               ray_dirs,
                                                                               r=self.object_bounding_sphere)

                net_values, steps = self.eval_sdf_inside_sphere(num_pixels,
                                                                lambda x: self.implicit_network(x, expression, include_color=False),
                                                                cam_loc,
                                                                ray_dirs,
                                                                mask_intersect,
                                                                sphere_intersections[..., 0],
                                                                sphere_intersections[..., 1],
                                                                pose_params=pose_params)

                variance = torch.tensor(vari).cuda()
                inv_s = torch.exp(variance * 10)
                prev_cdf = torch.sigmoid(net_values[:, :-1, 0] * inv_s)
                next_cdf = torch.sigmoid(net_values[:, 1:, 0] * inv_s)

                p = prev_cdf - next_cdf
                c = prev_cdf

                alpha = ((p + 1e-8) / (c + 1e-8)).clip(0.0, 1.0)

                weights = alpha * torch.cumprod(torch.cat([torch.ones([alpha.shape[0], 1], device='cuda'), 1. - alpha + 1e-7], dim=-1), dim=-1)[:, :-1]
                irrelevant_mask = weights < 1e-2
                irrelevant_mask = torch.zeros_like(irrelevant_mask)

                weights[irrelevant_mask] = 0
                net_values[:, :-1, :][irrelevant_mask, :] = 0

                dists = None

            # hierarchical sampling: given coarse sdf values perform importance sampling
            new_samples, z_samples = self.up_sample(cam_loc,
                                                    ray_dirs,
                                                    steps,
                                                    net_values[:, :, 0],
                                                    num_samples,
                                                    inv_s)#sphere_radius=pose_params[2])

            if use_SH:
                dists = z_samples[..., 1:] - z_samples[..., :-1]
                # dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
                dists = torch.cat([dists, dists[..., -1:]], -1)
                mid_z_vals = z_samples + dists * 0.5

                new_samples = cam_loc[:, None, :] + ray_dirs[0, :, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3

            n_rays = new_samples.shape[0]
            n_samples = new_samples.shape[1]
            sampled_points = new_samples
            net_values = []
            sdf_fun = lambda x: self.implicit_network(x, expression, include_color=True, return_grad=use_SH)
            for pnts in torch.split(new_samples.reshape(-1, 3), 25000, dim=0):
                if pose_params is not None:
                    pnts = pose_params[2] * pnts @ so3_exp_map(pose_params[0]).squeeze().T + pose_params[1]
                net_values.append(sdf_fun(pnts))
            net_values = torch.cat(net_values, dim=0).reshape(n_rays, n_samples, -1)
            points = sampled_points

            points = points.reshape(net_values.shape[0], net_values.shape[1], 3) # nrays x samples_per_ray x 3


            if use_SH:
                true_cos = (ray_dirs[0, :, None, :] * net_values[..., -3:]).sum(-1, keepdim=True)

                nphm_space_normals = net_values[..., -3:] / torch.norm(net_values[..., -3:], dim=-1, keepdim=True)
                world_space_normals_halfway = nphm_space_normals @ so3_exp_map(pose_params[0]).squeeze() # apply inverse wordl2model rotation
                world_space_normals = ((world_space_normals_halfway).view(-1, 3) @ w2c[:3, :3].T).view(num_pixels, -1, 3)  # TODO replace with einsum for speed
                shading = self.add_SHlight(world_space_normals, self.sh_coeffs)
            else:
                prev_sdf, next_sdf = net_values[:, :-1, 0], net_values[:, 1:, 0]
                prev_z_vals, next_z_vals = z_samples[:, :-1], z_samples[:, 1:]
                mid_sdf = (prev_sdf + next_sdf) * 0.5
                cos_val = (next_sdf - prev_sdf + 1e-8) / (next_z_vals - prev_z_vals + 1e-8)

                true_cos = cos_val.clip(-1e3, 0.0)
                world_space_normals = None

            variance = torch.tensor(neus_variance).cuda()
            inv_s = torch.exp(variance * 10)

            if use_SH:
                estimated_next_sdf = net_values[..., 0] + true_cos[..., 0] * dists * 0.5
                estimated_prev_sdf = net_values[..., 0] - true_cos[..., 0] * dists * 0.5
            else:
                estimated_next_sdf = mid_sdf + true_cos[..., 0] * dists * 0.5
                estimated_prev_sdf = mid_sdf - true_cos[..., 0] * dists * 0.5


            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            p = prev_cdf - next_cdf
            c = prev_cdf

            alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
            alpha = alpha.clip(0.0, 1.0)
            weights = alpha * torch.cumprod(
                torch.cat([torch.ones([alpha.shape[0], 1], device='cuda'), 1. - alpha + 1e-7], -1), -1)[:, :-1]

            #irrelevant_mask = weights < 1e-2
            #irrelevant_mask = torch.zeros_like(irrelevant_mask)
            #print(f'Percent of relevant samples: {(torch.numel(irrelevant_mask) - irrelevant_mask.sum()) / torch.numel(irrelevant_mask)}')

            weights_sum = weights.sum(dim=-1, keepdim=True)
            weighted_depth = (z_samples[:, :] * weights).sum(dim=-1)

            if use_SH:
                color = ((shading[:, :, :] * ((net_values[:, :, 1:4] + 1) / 2) - 0.5) * 2 * weights[:, :, None]).sum(dim=1)
            else:
                color = ((((net_values[:, :, 1:4] + 1) / 2) - 0.5) * 2 * weights[:, :, None]).sum(dim=1)

            if world_space_normals is None:
                normals = None
            else:
                normals = (world_space_normals * weights[:, :, None]).sum(dim=1)

            sdf_output = None,
            network_object_mask = None
            grad_theta = None
            dists = None
            rgb_values = color
        else:
            points = None
            rgb_values = None
            sdf_output=None
            network_object_mask = None
            object_mask = None
            dists = None
            grad_theta = None
            weights_sum = None
            weighted_depth = None
            normals = None

        output = {
            'points': points,
            'rgb_values': rgb_values,
            'sdf_output': sdf_output,
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'grad_theta': grad_theta,
            'dists': dists,
            'weights_sum': weights_sum,
            'weighted_depth': weighted_depth,
            'nphm_space_normals': normals,
        }

        return output


    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s, sphere_radius=1.0):
        """
        Up sampling give a fixed inv_s
        """
        #z_vals = z_vals[:, :-1]
        #sdf = sdf[:, :-1]

        batch_size, n_samples = z_vals.shape
        if rays_d.shape[1] != z_vals.shape[0]:
            print('hi')
        pts = rays_o[:, None, :] + rays_d[0, :, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < sphere_radius) | (radius[:, 1:] < sphere_radius)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf + 1e-8) / (next_z_vals - prev_z_vals + 1e-8)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device='cuda'), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-8) / (prev_cdf + 1e-8)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1], device='cuda'), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        #pl = pv.Plotter(
        #)
        #old_samples = rays_o[:, None, :] + rays_d[0, :, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        #pl.add_points(old_samples[:, :-1, :].reshape(-1, 3).detach().cpu().numpy(), scalars=weights.reshape(-1).detach().cpu().numpy())
        #pl.show()
        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        #z_samples_rnd = sample_pdf(z_vals, weights, n_importance, det=False).detach()

        new_samples = rays_o[:, None, :] + rays_d[0, :, None, :] * z_samples[..., :, None]  # n_rays, n_samples, 3
        #new_samples_rnd = rays_o[:, None, :] + rays_d[0, :, None, :] * z_samples_rnd[..., :, None]  # n_rays, n_samples, 3

        #pl = pv.Plotter()
        #pl.add_points(new_samples.reshape(-1, 3).detach().cpu().numpy())
        #pl.add_points(new_samples_rnd.reshape(-1, 3).detach().cpu().numpy(), color='red')
        #pl.show()

        return new_samples, z_samples

