import torch
import numpy as np
import os

from torch.utils.checkpoint import checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"

def phong_model(sdf, points, camera_position, phong_params, light_params):

    # TODO combine estimate_colors and estimate_normals to save one call to sdf
    colors = estimate_colors(sdf, points)
    normals = estimate_normals(sdf, points)
    view_dirs = points - camera_position
    light_dir_1 = light_params["light_dir_1"].repeat(points.shape[0], 1)
    light_dir_p = points - light_params["light_pos_p"].repeat(points.shape[0], 1)

    # Normalize all vectors
    normals = (normals.T / torch.norm(normals, dim=-1)).T
    light_dir_norm_1 = (light_dir_1.T / torch.norm(light_dir_1, dim=-1)).T
    light_dir_norm_p = (light_dir_p.T / torch.norm(light_dir_p, dim=-1)).T
    view_dir_norm = (view_dirs.T / torch.norm(view_dirs, dim=-1)).T

    # Ambient
    ambient = phong_params["ambient_coeff"] * light_params["amb_light_color"]
    ambient_refl = ambient.repeat(points.shape[0], 1)

    # Area light
    diffuse_1 = phong_params["diffuse_coeff"] * torch.clamp(torch.sum(-light_dir_norm_1 * normals, dim=-1), min=0.0) * \
                light_params["light_intensity_1"]# [N]
    diffuse_refl_1 = torch.matmul(diffuse_1.unsqueeze(1), light_params["light_color_1"].unsqueeze(0))  # [N, 3]
    reflect_dir_1 = light_dir_norm_1 + (
                2 * normals.T * torch.clamp(torch.sum(-light_dir_norm_1 * normals, dim=-1), min=0.0)).T
    specular_1 = phong_params["specular_coeff"] * torch.pow(
        torch.clamp(torch.sum(reflect_dir_1 * -view_dir_norm, dim=-1), min=0.0), phong_params["shininess"]) * \
                 light_params["light_intensity_1"] # [N]
    specular_refl_1 = torch.matmul(specular_1.unsqueeze(1), light_params["light_color_1"].unsqueeze(0))  # [N, 3]

    # Point light
    diffuse_p = phong_params["diffuse_coeff"] * torch.clamp(torch.sum(-light_dir_norm_p * normals, dim=-1), min=0.0) * \
                light_params["light_intensity_p"] # [N]
    diffuse_refl_p = torch.matmul(diffuse_p.unsqueeze(1), light_params["light_color_p"].unsqueeze(0))  # [N, 3]
    reflect_dir_p = light_dir_norm_p + (
                2 * normals.T * torch.clamp(torch.sum(-light_dir_norm_p * normals, dim=-1), min=0.0)).T
    specular_p = phong_params["specular_coeff"] * torch.pow(
        torch.clamp(torch.sum(reflect_dir_p * -view_dir_norm, dim=-1), min=0.0), phong_params["shininess"]) * \
                 light_params["light_intensity_p"]  # [N]
    specular_refl_p = torch.matmul(specular_p.unsqueeze(1), light_params["light_color_p"].unsqueeze(0))  # [N, 3]

    return ambient_refl + diffuse_refl_1 + specular_refl_1 + diffuse_refl_p + specular_refl_p, colors


def estimate_normals(sdf, points, epsilon=1e-3):
    sdf_inputs = torch.concat([points,
                               points + torch.tensor([epsilon, 0, 0]),
                               points + torch.tensor([0, epsilon, 0]),
                               points + torch.tensor([0, 0, epsilon])])

    sdf_values, _ = sdf(sdf_inputs)
    sdf_values = sdf_values.reshape(4, -1)

    # Calculate the gradient using finite differences
    gradient = sdf_values[1:] - sdf_values[0]

    # Normalize the gradient to obtain the estimated normal
    normal = gradient / torch.norm(gradient, p=2, dim=0)

    return normal.T

def estimate_colors(sdf, points):
    _, colors = sdf(points)
    colors = colors + 1
    colors /= 2
    return colors


def sphere_trace(sdf, camera_position, norm_directions, max_length):
    N = norm_directions.shape[0]
    positions = camera_position.unsqueeze(dim=0).repeat(N, 1)  # [N, 3]
    total_distances = torch.zeros(N)
    last_distances = torch.ones(N)

    for _ in range(20):
        # mask = torch.logical_and(total_distances < max_length, last_distances > 1e-3)
        not_reached_max_distance = total_distances < max_length
        not_hit_target = torch.abs(last_distances) > 1e-3
        mask = torch.logical_and(not_reached_max_distance, not_hit_target)
        if torch.all(torch.logical_not(mask)):
            break
        distances = sdf(positions[mask])
        steps = (norm_directions[mask].T * distances).T
        positions[mask] += steps
        total_distances[mask] += distances
        last_distances[mask] = distances

    # positions[total_distances > max_length] *= torch.nan
    return positions, total_distances < max_length


def acc_sphere_trace(sdf, init_position, norm_directions, max_length, scale=np.sqrt(2.), eps=1e-3, init_t=None):
    N = norm_directions.shape[0]
    if init_position.ndim > 1:
        positions = init_position
    else:
        positions = init_position.unsqueeze(dim=0).repeat(N, 1)  # [N, 3]

    r_last = torch.zeros(N)
    r_curr = torch.zeros(N)
    r_next = torch.ones(N)
    d_curr = torch.zeros(N)
    if init_t is not None:
        t = init_t
    else:
        t = torch.zeros(N)

    for i in range(15):
        not_reached_max_distance = t < max_length
        not_hit = torch.abs(r_next) > eps
        mask = torch.logical_and(not_reached_max_distance, not_hit)
        if torch.all(torch.logical_not(mask)):
            break

        d_curr[mask] = r_curr[mask] + scale * r_curr[mask] * torch.nan_to_num(
            (d_curr[mask] - r_last[mask] + r_curr[mask]) / (d_curr[mask] + r_last[mask] - r_curr[mask]))
        r_next[mask], _ = sdf(positions[mask] + ((t[mask] + d_curr[mask]) * norm_directions[mask].T).T)

        normal_tracing_mask = torch.abs(d_curr[mask]) > torch.abs(r_curr[mask]) + torch.abs(r_next[mask])
        if torch.any(normal_tracing_mask):
            d_curr[mask][normal_tracing_mask] = r_curr[mask][normal_tracing_mask]
            r_next[mask][normal_tracing_mask], _ = sdf(positions[mask][normal_tracing_mask] + (
                        (t[mask][normal_tracing_mask] + d_curr[mask][normal_tracing_mask]) * norm_directions[mask][
                    normal_tracing_mask].T).T)

        t[mask] += d_curr[mask]
        r_last[mask] = r_curr[mask]
        r_curr[mask] = r_next[mask]

    # hit_mask = torch.logical_and(t < max_length and r_next < eps)
    hit_mask = t < max_length
    hits = torch.zeros(N, 3)
    hits[hit_mask] = positions[hit_mask] + (t[hit_mask] * norm_directions[hit_mask].T).T
    return hits, hit_mask, t


def two_phase_tracing(sdf, camera_position, norm_directions, max_length, scale=np.sqrt(2.), eps=1e-3):
    N = norm_directions.shape[0]
    with torch.no_grad():
        hits_1, hit_mask_1, t_1 = acc_sphere_trace(sdf, camera_position, norm_directions, max_length, scale=2., eps=0.025)
    
    hits_2, hit_mask_2, t_2 = acc_sphere_trace(sdf, hits_1[hit_mask_1], norm_directions[hit_mask_1], 3., scale=np.sqrt(2.), eps=0.005)

    hit_mask = torch.zeros(N).bool()
    hit_mask[hit_mask_1] = hit_mask_2

    hits = torch.zeros(N, 3)
    hits[hit_mask] = hits_2[hit_mask_2]

    return hits, hit_mask


def render(model, lat_rep, camera_params, phong_params, light_params, color=True, mesh_path=None):

    def sdf(positions, chunk_size=10000):
    
        def get_sdf(nphm_input, lat_rep_in):
            #distance = model(nphm_input.to(device), lat_rep_in, None)[0].to("cpu")
            distance, color = checkpoint(model, *[nphm_input.to(device), *lat_rep_in])
            distance = distance.to("cpu")
            color = color.to("cpu")
            return distance.squeeze(), color.squeeze()
            
        nphm_input = torch.reshape(positions, (1, -1, 3))
        
        if nphm_input.shape[1] > chunk_size:
            chunked = torch.chunk(nphm_input, chunks=nphm_input.shape[1] // chunk_size, dim=1)
            distances, colors = zip(*[get_sdf(chunk, lat_rep) for chunk in chunked])
            return torch.cat(distances, dim=0), torch.cat(colors, dim=0)
        else:
            #distance = model(nphm_input.to(device), lat_rep_in.to(device).requires_grad_(True), None)[0].to("cpu")
            distance, color = get_sdf(nphm_input, lat_rep)
            return distance, color

    pu = camera_params["resolution_x"]
    pv = camera_params["resolution_y"]
    image = phong_params["background_color"].repeat(pu * pv, 1)

    angle_radians = torch.deg2rad_(torch.tensor(camera_params["camera_angle"]))
    camera = torch.tensor([torch.sin(angle_radians), 0, torch.cos(angle_radians)])
    camera_position = camera * (camera_params["camera_distance"] + camera_params["focal_length"]) / camera.norm()

    # Normalize the xy value of the current pixel [-0.5, 0.5]
    u_norms = ((torch.arange(pu) + 0.5) / pu - 0.5) * pu / pv
    v_norms = 0.5 - (torch.arange(pv) + 0.5) / pv

    # Calculate the ray directions for all pixels
    directions_unn = torch.cat(
        torch.meshgrid(u_norms, v_norms, torch.tensor(-camera_params["focal_length"]), indexing='ij'), dim=-1)
    directions_unn = directions_unn.reshape(
        (pu * pv, 3))  # [pu, pv, 3] --> [pu*pv, 3] (u1, v1, f)(u1, v2, f)...(u2, v1, f)...
    directions_unn = directions_unn

    # rotate about y-axis
    rotation_matrix = torch.tensor([[torch.cos(angle_radians), 0, torch.sin(angle_radians)],
                                    [0, 1, 0],
                                    [-torch.sin(angle_radians), 0, torch.cos(angle_radians)]])
    rotated_directions = torch.matmul(directions_unn, rotation_matrix.T)

    transposed_directions = rotated_directions.T  # transpose is necessary for normalization
    directions = (transposed_directions / transposed_directions.norm(dim=0)).T  # [pu*pv, 3]

    with torch.no_grad():
        # start close to head model to get useful sdf scores
        first_step_length = camera_params['focal_length'] + camera_params['camera_distance'] - 1
        N = directions.shape[0]
        starting_positions = camera_position.unsqueeze(dim=0).repeat(N, 1) + first_step_length * directions

        hits, hit_mask, _ = acc_sphere_trace(sdf, starting_positions, directions, camera_params['max_ray_length'], scale=np.sqrt(2.), eps=0.001)

        #hits_1, hit_mask_1, _ = acc_sphere_trace(sdf, starting_positions, directions, camera_params['max_ray_length'], scale=np.sqrt(2.), eps=0.025)
    
        #hits_2, hit_mask_2, _ = acc_sphere_trace(sdf, hits_1[hit_mask_1], directions[hit_mask_1], camera_params['max_ray_length'], scale=np.sqrt(2.), eps=0.001)
            

    #hit_mask = torch.zeros(N).bool()
    #hit_mask[hit_mask_1] = hit_mask_2
    #hit_mask = torch.zeros(N).bool()
    #hit_mask[hit_mask_1] = hit_mask_2

    #hits = torch.zeros(N, 3)
    #hits[hit_mask] = hits_2[hit_mask_2]
    #hits = torch.zeros(N, 3)
    #hits[hit_mask] = hits_2[hit_mask_2]

    max_number = 80000000
    phong_points = hits[hit_mask]
    if phong_points.shape[0] > max_number:
        with torch.no_grad():
            gradient_mask = torch.zeros(phong_points.shape[0])
            gradient_mask[:max_number] = 1
            gradient_mask = gradient_mask[torch.randperm(len(gradient_mask))].bool()
            no_gradient_mask = ~gradient_mask

            no_gradient_reflections, no_gradient_colors = phong_model(sdf, phong_points[no_gradient_mask, :], camera_position, phong_params, light_params)
        gradient_reflections, gradient_colors = phong_model(sdf, phong_points[gradient_mask, :], camera_position, phong_params, light_params)
        reflections = torch.zeros_like(phong_points).float() 
        colors = torch.zeros_like(phong_points).float()
        reflections[gradient_mask] = gradient_reflections
        reflections[no_gradient_mask] = no_gradient_reflections
        colors[gradient_mask] = gradient_colors
        colors[no_gradient_mask] = no_gradient_colors
    else:
        reflections, colors = phong_model(sdf, phong_points, camera_position, phong_params, light_params)
        

    # Assign a color for objects
    if color == True:
        image[hit_mask] = torch.mul(reflections, colors)
    else:
        image[hit_mask] = torch.mul(reflections, phong_params["object_color"].repeat(reflections.shape[0], 1))

    image = torch.clamp(image, min=0.0, max=1.0)
    nan_mask = torch.isnan(image)
    image = torch.where(nan_mask, torch.tensor(0.0), image)
    image = image.reshape(pu, pv, 3).transpose(0, 1)

    return image