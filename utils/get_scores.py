import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from facenet_pytorch import MTCNN, InceptionResnetV1
import pandas as pd

from utils.pipeline import get_image_clip_embedding, get_text_clip_embedding, clip_score
device = "cuda"

DINO_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
DINO_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
mtcnn = MTCNN(select_largest=True, device='cpu')
facenet = InceptionResnetV1(pretrained='vggface2').eval()

embeddings = torchvision.datasets.ImageFolder("../datasets/celebs", transform=lambda x: facenet(mtcnn(x).unsqueeze(0)))
embeddings.idx_to_class = {i:c for c, i in embeddings.class_to_idx.items()}
loader = torch.utils.data.DataLoader(embeddings)

####################### TOUCH ZONE #######################

lat_rep_path = '../notebooks/lat_rep_Audrey_Hepburn_1.pt'
prompt ='Audrey Hepburn'
gt_image_paths = ['../celebs/Audrey_Hepburn_1.png',
                  '../celebs/Audrey_Hepburn_2.png',
                  '../celebs/Audrey_Hepburn_3.png'] # square png image

##########################################################

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

camera_params_side= {
            "camera_distance": 0.21 * 2.57,
            "camera_angle_rho": 45.,
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

phong_params_no_color = {
        "ambient_coeff": 0.51,
        "diffuse_coeff": 0.75,
        "specular_coeff": 0.64,
        "shininess": 0.5,
        # Colors
        "object_color": torch.tensor([0.3, 0.3, 0.3]),
        "background_color": torch.tensor([1., 1., 1.])
    }

light_params_no_color = {
        "amb_light_color": torch.tensor([0.65, 0.65, 0.65]),
        # light 1
        "light_intensity_1": 1.42,
        "light_color_1": torch.tensor([1., 1., 1.]),
        "light_dir_1": torch.tensor([0., -0.4, -0.67]),
        # light p
        "light_intensity_p": 0.62,
        "light_color_p": torch.tensor([1., 1., 1.]),
        "light_pos_p": torch.tensor([1.19, -1.27, 2.24])
    }

def get_image_DINO_embedding(image):
    input = DINO_processor(images=image, return_tensors="pt", do_rescale=False).to(device)
    output = DINO_model(**input)
    CLS_token = output.last_hidden_state # [1, 257, 768]
    DINO_embedding  = CLS_token.mean(dim=1) # [1, 768]
    DINO_embedding /= DINO_embedding.norm(dim=-1, keepdim=True)
    return DINO_embedding

def dino_score(DINO_embedding, gt_embedding):
    DINO_similarity = 100 * torch.matmul(DINO_embedding, gt_embedding.T)
    return DINO_similarity

def get_facenet_distance_to_ds(lat_rep_render, prompt):
    lat_cropped = mtcnn(torchvision.transforms.functional.to_pil_image(lat_rep_render.permute(2, 0, 1)))
    lat_embedding = facenet(lat_cropped.unsqueeze(0)) 
    
    names = []
    distances = []
    for x, y in loader:
        names.append(embeddings.idx_to_class[int(y[0])])
        distances.append((lat_embedding - x).norm().detach().cpu())
    
    scores = pd.Series(distances, index=names).groupby(names).mean()
    prompt_score = scores[prompt].mean()
    others_score = scores.drop(prompt).mean()
        
    return scores, prompt_score, others_score

def get_scores(lat_rep, prompt, gt_image_paths):

    normalized_CLIP_embedding_text = get_text_clip_embedding(prompt)

    normalized_CLIP_embedding_image_front_c, image_front_c = get_image_clip_embedding(lat_rep, camera_params_front, phong_params_color, light_params_color, with_app_grad=False, color=True)
    normalized_CLIP_embedding_image_side_c, image_side_c = get_image_clip_embedding(lat_rep, camera_params_side, phong_params_color, light_params_color, with_app_grad=False, color=True)
    normalized_CLIP_embedding_image_front_no_c, image_front_no_c = get_image_clip_embedding(lat_rep, camera_params_front, phong_params_no_color, light_params_no_color, with_app_grad=False, color=False)
    normalized_CLIP_embedding_image_side_no_c, image_side_no_c = get_image_clip_embedding(lat_rep, camera_params_side, phong_params_no_color, light_params_no_color, with_app_grad=False, color=False)
    
    CLIP_score_front_c = clip_score(normalized_CLIP_embedding_image_front_c, normalized_CLIP_embedding_text)
    CLIP_score_side_c = clip_score(normalized_CLIP_embedding_image_side_c, normalized_CLIP_embedding_text)
    CLIP_score_front_no_c = clip_score(normalized_CLIP_embedding_image_front_no_c, normalized_CLIP_embedding_text)
    CLIP_score_side_no_c = clip_score(normalized_CLIP_embedding_image_side_no_c, normalized_CLIP_embedding_text)
    
    get_facenet_distance_to_ds(image_front_c)

    normalized_DINO_embedding_image_front_c = get_image_DINO_embedding(image_front_c)
    normalized_DINO_embedding_image_side_c = get_image_DINO_embedding(image_side_c)
    normalized_DINO_embedding_image_front_no_c = get_image_DINO_embedding(image_front_no_c)
    normalized_DINO_embedding_image_side_no_c = get_image_DINO_embedding(image_side_no_c)

    gt_DINO = []
    DINO_front_c = []
    DINO_side_c = []
    DINO_front_no_c = []
    DINO_side_no_c = []

    for image_path in gt_image_paths:
        image_pil = Image.open(image_path)
        image_pil_rgb = image_pil.convert('RGB')
        image_numpy = np.array(image_pil_rgb)
        gt_image = torch.tensor(image_numpy/255)

        normalized_DINO_embedding_image_gt = get_image_DINO_embedding(gt_image)
        gt_DINO.append(normalized_DINO_embedding_image_gt)

        DINO_score_front_c = dino_score(normalized_DINO_embedding_image_front_c, normalized_DINO_embedding_image_gt)
        DINO_front_c.append(DINO_score_front_c)
        DINO_score_side_c = dino_score(normalized_DINO_embedding_image_side_c, normalized_DINO_embedding_image_gt)
        DINO_side_c.append(DINO_score_side_c)
        DINO_score_front_no_c = dino_score(normalized_DINO_embedding_image_front_no_c, normalized_DINO_embedding_image_gt)
        DINO_front_no_c.append(DINO_score_front_no_c)
        DINO_score_side_no_c = dino_score(normalized_DINO_embedding_image_side_no_c, normalized_DINO_embedding_image_gt)
        DINO_side_no_c.append(DINO_score_side_no_c)
    
    dino_scores_front_c = torch.tensor(DINO_front_c)
    dino_scores_side_c = torch.tensor(DINO_side_c)
    dino_scores_front_no_c = torch.tensor(DINO_front_no_c)
    dino_scores_side_no_c = torch.tensor(DINO_side_no_c)

    dino_front_c = dino_scores_front_c.mean()
    dino_side_c = dino_scores_side_c.mean()
    dino_front_no_c = dino_scores_front_no_c.mean()
    dino_side_no_c = dino_scores_side_no_c.mean()

    data = {
        'Condition': ['Front view with color', 'Side view with color', 'Front view without color', 'Side view without color'],
        'CLIP Score': [CLIP_score_front_c.detach().cpu().numpy(), CLIP_score_side_c.detach().cpu().numpy(), CLIP_score_front_no_c.detach().cpu().numpy(), CLIP_score_side_no_c.detach().cpu().numpy()],
        'DINO Score': [dino_front_c.detach().cpu().numpy(), dino_side_c.detach().cpu().numpy(), dino_front_no_c.detach().cpu().numpy(), dino_side_no_c.detach().cpu().numpy()]
    }

    df = pd.DataFrame(data)

    all_clip_scores = np.array(data['CLIP Score'])
    all_dino_scores = np.array(data['DINO Score'])

    mean_clip_scores = np.mean(all_clip_scores, axis=0)
    mean_dino_scores = np.mean(all_dino_scores, axis=0)

    print('###########################################################################')
    print(f'results for {prompt}')
    print(df)

    print(f'Mean CLIP Scores: {mean_clip_scores}')
    print(f'Mean DINO Scores: {mean_dino_scores}')

    dino_scores = []
    n = len(gt_DINO)
    for i in range(n):
        dino_emb_1 = gt_DINO[i]
        for j in range((i+1), n):
            dino_emb_2 = gt_DINO[j]
            dino_sim = dino_score(dino_emb_1, dino_emb_2)
            dino_scores.append(dino_sim)

    dinos = torch.tensor(dino_scores)
    gt_dino_mean = dinos.mean()

    print(f'The mean DINO similarity of {n} gt images is: {gt_dino_mean}')

lat_rep = torch.load(lat_rep_path)
lat_rep = [tensor.to(device) for tensor in lat_rep]

with torch.no_grad():
    get_scores(lat_rep, prompt, gt_image_paths)
