import torch
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"

def phong_model(sdf, points, camera_position, phong_params, light_params, mesh_path, index_tri=None):
    # Option 1: Use SDF
    normals = estimate_normals(sdf, points)
    # Option 2: Use Mesh
    # normals = mesh_normals(mesh_path, index_tri)
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
                light_params["light_intensity_1"]  # [N]
    diffuse_refl_1 = torch.matmul(diffuse_1.unsqueeze(1), light_params["light_color_1"].unsqueeze(0))  # [N, 3]
    reflect_dir_1 = light_dir_norm_1 + (
                2 * normals.T * torch.clamp(torch.sum(-light_dir_norm_1 * normals, dim=-1), min=0.0)).T
    specular_1 = phong_params["specular_coeff"] * torch.pow(
        torch.clamp(torch.sum(reflect_dir_1 * -view_dir_norm, dim=-1), min=0.0), phong_params["shininess"]) * \
                 light_params["light_intensity_1"]  # [N]
    specular_refl_1 = torch.matmul(specular_1.unsqueeze(1), light_params["light_color_1"].unsqueeze(0))  # [N, 3]

    # Point light
    diffuse_p = phong_params["diffuse_coeff"] * torch.clamp(torch.sum(-light_dir_norm_p * normals, dim=-1), min=0.0) * \
                light_params["light_intensity_p"]  # [N]
    diffuse_refl_p = torch.matmul(diffuse_p.unsqueeze(1), light_params["light_color_p"].unsqueeze(0))  # [N, 3]
    reflect_dir_p = light_dir_norm_p + (
                2 * normals.T * torch.clamp(torch.sum(-light_dir_norm_p * normals, dim=-1), min=0.0)).T
    specular_p = phong_params["specular_coeff"] * torch.pow(
        torch.clamp(torch.sum(reflect_dir_p * -view_dir_norm, dim=-1), min=0.0), phong_params["shininess"]) * \
                 light_params["light_intensity_p"]  # [N]
    specular_refl_p = torch.matmul(specular_p.unsqueeze(1), light_params["light_color_p"].unsqueeze(0))  # [N, 3]

    return ambient_refl + diffuse_refl_1 + specular_refl_1 + diffuse_refl_p + specular_refl_p


def estimate_normals(sdf, points, epsilon=1e-3):
    sdf_inputs = torch.concat([points,
                               points + torch.tensor([epsilon, 0, 0]),
                               points + torch.tensor([0, epsilon, 0]),
                               points + torch.tensor([0, 0, epsilon])])

    sdf_values = sdf(sdf_inputs).reshape(4, -1)

    # Calculate the gradient using finite differences
    gradient = sdf_values[1:] - sdf_values[0]

    # Normalize the gradient to obtain the estimated normal
    normal = gradient / torch.norm(gradient, p=2, dim=0)

    return normal.T


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

    sdf_calls = torch.zeros(N)

    for i in range(15):
        not_reached_max_distance = t < max_length
        not_hit = torch.abs(r_next) > eps
        mask = torch.logical_and(not_reached_max_distance, not_hit)
        if torch.all(torch.logical_not(mask)):
            break

        d_curr[mask] = r_curr[mask] + scale * r_curr[mask] * torch.nan_to_num(
            (d_curr[mask] - r_last[mask] + r_curr[mask]) / (d_curr[mask] + r_last[mask] - r_curr[mask]))
        r_next[mask] = sdf(positions[mask] + ((t[mask] + d_curr[mask]) * norm_directions[mask].T).T)

        normal_tracing_mask = torch.abs(d_curr[mask]) > torch.abs(r_curr[mask]) + torch.abs(r_next[mask])
        if torch.any(normal_tracing_mask):
            d_curr[mask][normal_tracing_mask] = r_curr[mask][normal_tracing_mask]
            r_next[mask][normal_tracing_mask] = sdf(positions[mask][normal_tracing_mask] + (
                        (t[mask][normal_tracing_mask] + d_curr[mask][normal_tracing_mask]) * norm_directions[mask][
                    normal_tracing_mask].T).T)
            sdf_calls[mask][normal_tracing_mask] += 1

        t[mask] += d_curr[mask]
        r_last[mask] = r_curr[mask]
        r_curr[mask] = r_next[mask]

        sdf_calls[mask] += 1

    # hit_mask = torch.logical_and(t < max_length and r_next < eps)
    hit_mask = t < max_length
    hits = torch.zeros(N, 3)
    hits[hit_mask] = positions[hit_mask] + (t[hit_mask] * norm_directions[hit_mask].T).T
    return hits, hit_mask, sdf_calls, t


def two_phase_tracing(sdf, camera_position, norm_directions, max_length, scale=np.sqrt(2.), eps=1e-3):
    N = norm_directions.shape[0]
    with torch.no_grad():
        hits_1, hit_mask_1, sdf_calls_1, t_1 = acc_sphere_trace(sdf, camera_position, norm_directions, max_length,
                                                                scale=2., eps=0.025)
    hits_2, hit_mask_2, sdf_calls_2, t_2 = acc_sphere_trace(sdf, hits_1[hit_mask_1], norm_directions[hit_mask_1], 3.,
                                                            scale=np.sqrt(2.), eps=0.005)

    hit_mask = torch.zeros(N).bool()
    hit_mask[hit_mask_1] = hit_mask_2

    hits = torch.zeros(N, 3)
    hits[hit_mask] = hits_2[hit_mask_2]

    with torch.no_grad():
        sdf_calls = torch.zeros_like(sdf_calls_1)
        sdf_calls[hit_mask_1] += sdf_calls_2

    return hits, hit_mask


def render(model, lat_rep, camera_params, phong_params, light_params, mesh_path=None):
    def sdf(positions, max_number=10000):
        nphm_input = torch.reshape(positions, (1, -1, 3))

        lat_rep_in = torch.reshape(lat_rep, (1, 1, -1))

        if nphm_input.shape[1] > max_number:
            chunked = torch.split(nphm_input, max_number, dim=1)
            distances = []
            for chunk in chunked:
                distance = model(chunk.to(device), lat_rep_in.to(device), None)[0].to("cpu")
                distances.append(distance)
            return torch.cat(distances, dim=1).squeeze()
        else:
            distance = model(nphm_input.to(device), lat_rep_in.to(device), None)[0].to("cpu")
            return distance.squeeze()

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

    # Option 1: Use SDF
    hit_positions, hit_mask = two_phase_tracing(sdf, camera_position, directions, camera_params['max_ray_length'])
    # Option 2: Use Mesh
    # intersections, hit_mask, index_tri = mesh_trace(mesh_path, camera_position, directions)

    # Option 1: Use SDF
    reflections = phong_model(sdf, hit_positions[hit_mask], camera_position, phong_params, light_params, mesh_path)
    # Option 2: Use Mesh
    # reflections = phong_model(sdf, intersections, camera_position, phong_params, light_params, mesh_path, index_tri) # mesh alternative

    # Assign a color for objects
    image[hit_mask] = torch.mul(reflections, phong_params["object_color"].repeat(reflections.shape[0], 1))
    image = torch.clamp(image, max=1.0)
    image = image.reshape(pu, pv, 3).transpose(0, 1)

    return image