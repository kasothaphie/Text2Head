import torch



def phong_model(normals, light_dirs, view_dirs, ambient_coeff, diffuse_coeff, specular_coeff, shininess):
    # Normalize all vectors
    light_dirs_norm = (light_dirs.T / torch.norm(light_dirs, dim=-1)).T
    view_dirs_norm = (view_dirs.T / torch.norm(view_dirs, dim=-1)).T
    
    ambient = ambient_coeff
    diffuses = diffuse_coeff * torch.clamp(torch.sum(light_dirs_norm * normals, dim=-1), min=0.0)
    reflect_dirs = light_dirs - (2 * normals.T * torch.clamp(torch.sum(light_dirs_norm * normals, dim=-1), min=0.0)).T
    speculars = specular_coeff * torch.pow(torch.clamp(torch.sum(reflect_dirs * view_dirs_norm, dim=-1), min=0.0), shininess)
    
    return ambient + diffuses + speculars

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
    positions = camera_position.unsqueeze(dim=0).repeat(N, 1)
    total_distances = torch.zeros(N)
    last_distances = torch.ones(N)
    
    for _ in range(20):
        #mask = torch.logical_and(total_distances < max_length, last_distances > 1e-3)
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
        
    #positions[total_distances > max_length] *= torch.nan
        
    return positions, total_distances < max_length
    

def render(sdf, px, camera_position, light_position, ambient_coeff, diffuse_coeff, specular_coeff, shininess, max_ray_length=4.):
    image = torch.zeros((px * px, 3))
    # Normalize the xy value of the current pixel [-1, 1]
    u_norms = 2.0 * (torch.arange(px) + 0.5) / px - 1.0
    v_norms = 1.0 - 2.0 * (torch.arange(px) + 0.5) / px

    # Calculate the ray directions for all pixelx
    directions_unn = torch.stack(torch.meshgrid(u_norms, v_norms, torch.tensor(-3.0))).reshape((3, -1))
    directions = (directions_unn / directions_unn.norm(dim=0)).T
        
    # Perform sphere tracing
    hit_positions, hit_mask = sphere_trace(sdf, camera_position, directions, max_ray_length)
    
    # Color the pixel based on whether the ray hits an object
    normals = estimate_normals(sdf, hit_positions[hit_mask])
    light_dirs = - (hit_positions[hit_mask] - light_position) # umdrehen, damit L*V >0
    view_dirs = - (camera_position - hit_positions[hit_mask]) # umdrehen, damit dot product nicht kleienr null?
    reflections = phong_model(normals, light_dirs, view_dirs, ambient_coeff, diffuse_coeff, specular_coeff, shininess)
    
    # Assign a color for objects
    image[hit_mask] = torch.matmul(reflections.unsqueeze(1), torch.tensor([1.0, 1.0, 1.0]).unsqueeze(0))
    image = image.reshape(px, px, 3).transpose(0, 1)
            
    return image