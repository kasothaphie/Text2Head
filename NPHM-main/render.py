import torch


def phong_model(sdf, points, camera_position, ambient_coeff, diffuse_coeff, specular_coeff, shininess):
    # define
    amb_light_color = torch.tensor([0.15, 0, 0])
    light_color = torch.tensor([1.0, 1.0, 1.0])
    # point light
    light_intensity_1 = 1.
    light_position_1 = torch.tensor([1., 1., 1.])
    # area light
    light_intensity_2 = 0.5
    light_dir_2 = torch.tensor([0., 0., -1.])

    normals = estimate_normals(sdf, points)
    view_dirs = points - camera_position
    light_dir_1 = points - light_position_1
    light_dir_2 = light_dir_2.repeat(points.shape[0], 1)

    # Normalize all vectors
    light_dir_norm_1 = (light_dir_1.T / torch.norm(light_dir_1, dim=-1)).T
    light_dir_norm_2 = (light_dir_2.T / torch.norm(light_dir_2, dim=-1)).T
    view_dir_norm = (view_dirs.T / torch.norm(view_dirs, dim=-1)).T

    ambient = ambient_coeff * amb_light_color  # TODO: is * amb_light_color necessary???
    ambient_refl = ambient.repeat(points.shape[0], 1)

    # point light
    diffuse_1 = diffuse_coeff * torch.clamp(torch.sum(-light_dir_norm_1 * normals, dim=-1),
                                            min=0.0) * light_intensity_1  # [N]
    diffuse_refl_1 = torch.matmul(diffuse_1.unsqueeze(1), light_color.unsqueeze(0))  # [N, 3]
    reflect_dir_1 = light_dir_norm_1 - (
                2 * normals.T * torch.clamp(torch.sum(-light_dir_norm_1 * normals, dim=-1), min=0.0)).T
    specular_1 = specular_coeff * torch.pow(torch.clamp(torch.sum(reflect_dir_1 * -view_dir_norm, dim=-1), min=0.0),
                                            shininess) * light_intensity_1  # [N]
    specular_refl_1 = torch.matmul(specular_1.unsqueeze(1), light_color.unsqueeze(0))  # [N, 3]

    # area light
    diffuse_2 = diffuse_coeff * torch.clamp(torch.sum(-light_dir_norm_2 * normals, dim=-1),
                                            min=0.0) * light_intensity_2  # [N]
    diffuse_refl_2 = torch.matmul(diffuse_2.unsqueeze(1), light_color.unsqueeze(0))  # [N, 3]
    reflect_dir_2 = light_dir_norm_2 - (
                2 * normals.T * torch.clamp(torch.sum(-light_dir_norm_2 * normals, dim=-1), min=0.0)).T
    specular_2 = specular_coeff * torch.pow(torch.clamp(torch.sum(reflect_dir_2 * -view_dir_norm, dim=-1), min=0.0),
                                            shininess) * light_intensity_2  # [N]
    specular_refl_2 = torch.matmul(specular_2.unsqueeze(1), light_color.unsqueeze(0))  # [N, 3]

    return ambient_refl + diffuse_refl_1 + specular_refl_1 + diffuse_refl_2 + specular_refl_2


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


def render(sdf, pu, pv, camera_distance, camera_angle, ambient_coeff, diffuse_coeff, specular_coeff, shininess,
           focal_length, max_ray_length=4.):
    '''

    Parameters
    ----------
    sdf
    pu: resolution in x-direction
    pv: resolution in y-direction
    camera_distance: distance of camera to origin
    camera_angle: angle around y-axis between camera position and z-axis in degree
    ambient_coeff
    diffuse_coeff
    specular_coeff
    shininess
    focal_length
    max_ray_length

    Returns
    -------

    '''
    object_color = torch.tensor([0.61, 0.61, 0.61])
    background_color = torch.tensor([0.15, 0, 0])

    image = background_color.repeat(pu * pv, 1)

    angle_radians = torch.deg2rad_(torch.tensor(camera_angle))
    camera = torch.tensor([torch.sin(angle_radians), 0, torch.cos(angle_radians)])
    camera_position = camera * camera_distance / camera.norm()

    # Normalize the xy value of the current pixel [-1, 1]
    u_norms = 2.0 * (torch.arange(pu) + 0.5) / pu - 1.0
    v_norms = (1.0 - 2.0 * (torch.arange(pv) + 0.5) / pv) * pv / pu

    # Calculate the ray directions for all pixels
    directions_unn = torch.cat(torch.meshgrid(u_norms, v_norms, torch.tensor(-focal_length)), dim=-1)
    directions_unn = directions_unn.reshape(
        (pu * pv, 3))  # [pu, pv, 3] --> [pu*pv, 3] (u1, v1, f)(u1, v2, f)...(u2, v1, f)...

    # rotate about y-axis
    rotation_matrix = torch.tensor([[torch.cos(angle_radians), 0, torch.sin(angle_radians)],
                                    [0, 1, 0],
                                    [-torch.sin(angle_radians), 0, torch.cos(angle_radians)]])
    rotated_directions = torch.matmul(directions_unn, rotation_matrix.T)

    transposed_directions = rotated_directions.T  # transpose is necessary for normalization
    directions = (transposed_directions / transposed_directions.norm(dim=0)).T  # [pu*pv, 3]

    # Perform sphere tracing
    hit_positions, hit_mask = sphere_trace(sdf, camera_position, directions, max_ray_length)

    # Color the pixel based on whether the ray hits an object
    reflections = phong_model(sdf, hit_positions[hit_mask], camera_position, ambient_coeff, diffuse_coeff,
                              specular_coeff, shininess)

    # Assign a color for objects
    image[hit_mask] = torch.mul(reflections, object_color.repeat(reflections.shape[0], 1))
    image = torch.clamp(image, max=1.0)
    image = image.reshape(pu, pv, 3).transpose(0, 1)

    return image