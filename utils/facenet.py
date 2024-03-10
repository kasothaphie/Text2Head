from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torchvision
import pandas as pd
from utils.pipeline import forward

mtcnn = MTCNN(select_largest=True, device='cpu')
facenet = InceptionResnetV1(pretrained='vggface2').eval()

embeddings = torchvision.datasets.ImageFolder("../datasets/celebs", transform=lambda x: facenet(mtcnn(x).unsqueeze(0)))
embeddings.idx_to_class = {i:c for c, i in embeddings.class_to_idx.items()}
loader = torch.utils.data.DataLoader(embeddings)

resolution = 400

camera_params_front = {
            "camera_distance": 0.21 * 2.57,
            "camera_angle_rho": 0.,
            "camera_angle_theta": 0.,
            "focal_length": 2.57,
            "max_ray_length": 3.5,
            # Image
            "resolution_y": resolution,
            "resolution_x": resolution
        }

phong_params_color = {
            "ambient_coeff": 0.32,
            "diffuse_coeff": 0.85,
            "specular_coeff": 0.34,
            "shininess": 25,
            # Colors
            "background_color": torch.tensor([1., 1., 1.])
        }

light_params_color = {
            "amb_light_color": torch.tensor([0.65, 0.65, 0.65]),
            # light 1
            "light_intensity_1": 1.69,
            "light_color_1": torch.tensor([1., 1., 1.]),
            "light_dir_1": torch.tensor([0, -0.18, -0.8]),
            # light p
            "light_intensity_p": 0.52,
            "light_color_p": torch.tensor([1., 1., 1.]),
            "light_pos_p": torch.tensor([0.17, 2.77, -2.25])
    }

def get_facenet_distance_to_ds(lat_rep):
    lat_rep = [lat.cuda() for lat in lat_rep]
    with torch.no_grad():
        lat_rep_render = forward(lat_rep, "", camera_params_front, phong_params_color, light_params_color, False, True)[4]
    
    lat_cropped = mtcnn(torchvision.transforms.functional.to_pil_image(lat_rep_render.permute(2, 0, 1)))
    lat_embedding = facenet(lat_cropped.unsqueeze(0)) 
    
    names = []
    distances = []
    for x, y in loader:
        names.append(embeddings.idx_to_class[int(y[0])])
        distances.append((lat_embedding - x).norm().detach().cpu())
        
    return pd.Series(distances, index=names)