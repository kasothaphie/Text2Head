import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from facenet_pytorch import MTCNN, InceptionResnetV1
import pandas as pd
import pickle
import os

from utils.pipeline import get_image_clip_embedding, get_text_clip_embedding, clip_score, CLIP_model, CLIP_preprocess
device = "cuda"

dataset_path = "../datasets/celebs/"

DINO_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
DINO_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
mtcnn = MTCNN(select_largest=True, device='cpu')
facenet = InceptionResnetV1(pretrained='vggface2').eval()

embeddings = torchvision.datasets.ImageFolder(dataset_path, transform=lambda x: facenet(mtcnn(x).unsqueeze(0)))
embeddings.idx_to_class = {i:c for c, i in embeddings.class_to_idx.items()}
loader = torch.utils.data.DataLoader(embeddings)

####################### TOUCH ZONE #######################

#lat_rep_path = './latents/color/ours_Chris_Rock_0'
#prompt ='Chris Rock'
#gt_image_paths = [f'./datasets/celebs/{prompt}/0.jpeg',
#                  f'./datasets/celebs/{prompt}/1.jpg',
#                  f'./datasets/celebs/{prompt}/2.jpg'] # square png image

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
    
    scores = pd.Series(distances, index=names)
    prompt_score = scores[prompt].mean()
    others_score = scores.drop(prompt).mean()
        
    return scores.groupby(names).mean(), prompt_score, others_score

# e.g. "Thomas Mueller": lat_rep
def get_scores_for_dict(lat_dict: dict):
    result_dict = {}
    for name in  lat_dict.keys():
        lat_rep = lat_dict[name]
        with torch.no_grad():
            scores = get_scores(lat_rep, name)
        result_dict[name] = scores
    return result_dict

lat_mean = [
    torch.zeros(2176, device=device),
    torch.zeros(100, device=device),
    torch.zeros(2176, device=device)
]

lat_mean_CLIP_embedding_image_front_c, lat_mean_image_front_c = get_image_clip_embedding(lat_mean, camera_params_front, phong_params_color, light_params_color, with_app_grad=False, color=True)
lat_mean_CLIP_embedding_image_side_c, lat_mean_image_side_c = get_image_clip_embedding(lat_mean, camera_params_side, phong_params_color, light_params_color, with_app_grad=False, color=True)
lat_mean_CLIP_embedding_image_front_no_c, lat_mean_image_front_no_c = get_image_clip_embedding(lat_mean, camera_params_front, phong_params_no_color, light_params_no_color, with_app_grad=False, color=False)
lat_mean_CLIP_embedding_image_side_no_c, lat_mean_image_side_no_c = get_image_clip_embedding(lat_mean, camera_params_side, phong_params_no_color, light_params_no_color, with_app_grad=False, color=False)

lat_mean_normalized_DINO_embedding_image_front_c = get_image_DINO_embedding(lat_mean_image_front_c)
lat_mean_normalized_DINO_embedding_image_side_c = get_image_DINO_embedding(lat_mean_image_side_c)
lat_mean_normalized_DINO_embedding_image_front_no_c = get_image_DINO_embedding(lat_mean_image_front_no_c)
lat_mean_normalized_DINO_embedding_image_side_no_c = get_image_DINO_embedding(lat_mean_image_side_no_c)

def get_scores(lat_rep, prompt):
    lat_rep = [tensor.to(device) for tensor in lat_rep]

    normalized_CLIP_embedding_text = get_text_clip_embedding(prompt)

    normalized_CLIP_embedding_image_front_c, image_front_c = get_image_clip_embedding(lat_rep, camera_params_front, phong_params_color, light_params_color, with_app_grad=False, color=True)
    normalized_CLIP_embedding_image_side_c, image_side_c = get_image_clip_embedding(lat_rep, camera_params_side, phong_params_color, light_params_color, with_app_grad=False, color=True)
    normalized_CLIP_embedding_image_front_no_c, image_front_no_c = get_image_clip_embedding(lat_rep, camera_params_front, phong_params_no_color, light_params_no_color, with_app_grad=False, color=False)
    normalized_CLIP_embedding_image_side_no_c, image_side_no_c = get_image_clip_embedding(lat_rep, camera_params_side, phong_params_no_color, light_params_no_color, with_app_grad=False, color=False)
    
    CLIP_score_front_c = clip_score(normalized_CLIP_embedding_image_front_c, normalized_CLIP_embedding_text)
    CLIP_score_side_c = clip_score(normalized_CLIP_embedding_image_side_c, normalized_CLIP_embedding_text)
    CLIP_score_front_no_c = clip_score(normalized_CLIP_embedding_image_front_no_c, normalized_CLIP_embedding_text)
    CLIP_score_side_no_c = clip_score(normalized_CLIP_embedding_image_side_no_c, normalized_CLIP_embedding_text)
    
    lat_mean_CLIP_score_front_c = clip_score(lat_mean_CLIP_embedding_image_front_c, normalized_CLIP_embedding_text)
    lat_mean_CLIP_score_side_c = clip_score(lat_mean_CLIP_embedding_image_side_c, normalized_CLIP_embedding_text)
    lat_mean_CLIP_score_front_no_c = clip_score(lat_mean_CLIP_embedding_image_front_no_c, normalized_CLIP_embedding_text)
    lat_mean_CLIP_score_side_no_c = clip_score(lat_mean_CLIP_embedding_image_side_no_c, normalized_CLIP_embedding_text)
    
    facenet_all_scores, facenet_prompt_score, facenet_rest_score = get_facenet_distance_to_ds(image_front_c, prompt)
    lat_mean_facenet_all_scores, lat_mean_facenet_prompt_score, lat_mean_facenet_rest_score = get_facenet_distance_to_ds(lat_mean_image_front_c, prompt)

    normalized_DINO_embedding_image_front_c = get_image_DINO_embedding(image_front_c)
    normalized_DINO_embedding_image_side_c = get_image_DINO_embedding(image_side_c)
    normalized_DINO_embedding_image_front_no_c = get_image_DINO_embedding(image_front_no_c)
    normalized_DINO_embedding_image_side_no_c = get_image_DINO_embedding(image_side_no_c)

    gt_DINO = []
    gt_CLIP = []
    DINO_front_c = []
    DINO_side_c = []
    DINO_front_no_c = []
    DINO_side_no_c = []
    
    lat_mean_DINO_front_c = []
    lat_mean_DINO_side_c = []
    lat_mean_DINO_front_no_c = []
    lat_mean_DINO_side_no_c = []

    prompt_path = dataset_path + prompt
    for filename in os.listdir(prompt_path):
        image_path = os.path.join(prompt_path, filename)
        
        image_pil = Image.open(image_path)
        image_pil_rgb = image_pil.convert('RGB')
        image_numpy = np.array(image_pil_rgb)
        gt_image = torch.tensor(image_numpy/255)
        
        image_preprocessed = CLIP_preprocess(image_pil).unsqueeze(0)
        CLIP_embedding_image = CLIP_model.encode_image(image_preprocessed.cuda()).cpu() # [1, 512]
        normalized_CLIP_embedding_image = CLIP_embedding_image / CLIP_embedding_image.norm(dim=-1, keepdim=True)
        
        gt_clip = clip_score(normalized_CLIP_embedding_image, normalized_CLIP_embedding_text)
        gt_CLIP.append(gt_clip)

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
        
        lat_mean_DINO_score_front_c = dino_score(lat_mean_normalized_DINO_embedding_image_front_c, normalized_DINO_embedding_image_gt)
        lat_mean_DINO_front_c.append(lat_mean_DINO_score_front_c)
        lat_mean_DINO_score_side_c = dino_score(lat_mean_normalized_DINO_embedding_image_side_c, normalized_DINO_embedding_image_gt)
        lat_mean_DINO_side_c.append(lat_mean_DINO_score_side_c)
        lat_mean_DINO_score_front_no_c = dino_score(lat_mean_normalized_DINO_embedding_image_front_no_c, normalized_DINO_embedding_image_gt)
        lat_mean_DINO_front_no_c.append(lat_mean_DINO_score_front_no_c)
        lat_mean_DINO_score_side_no_c = dino_score(lat_mean_normalized_DINO_embedding_image_side_no_c, normalized_DINO_embedding_image_gt)
        lat_mean_DINO_side_no_c.append(lat_mean_DINO_score_side_no_c)
    
    dino_scores_front_c = torch.tensor(DINO_front_c)
    dino_scores_side_c = torch.tensor(DINO_side_c)
    dino_scores_front_no_c = torch.tensor(DINO_front_no_c)
    dino_scores_side_no_c = torch.tensor(DINO_side_no_c)
    
    lat_mean_dino_scores_front_c = torch.tensor(lat_mean_DINO_front_c)
    lat_mean_dino_scores_side_c = torch.tensor(lat_mean_DINO_side_c)
    lat_mean_dino_scores_front_no_c = torch.tensor(lat_mean_DINO_front_no_c)
    lat_mean_dino_scores_side_no_c = torch.tensor(lat_mean_DINO_side_no_c)

    dino_front_c = dino_scores_front_c.mean()
    dino_side_c = dino_scores_side_c.mean()
    dino_front_no_c = dino_scores_front_no_c.mean()
    dino_side_no_c = dino_scores_side_no_c.mean()
    
    lat_mean_dino_front_c = lat_mean_dino_scores_front_c.mean()
    lat_mean_dino_side_c = lat_mean_dino_scores_side_c.mean()
    lat_mean_dino_front_no_c = lat_mean_dino_scores_front_no_c.mean()
    lat_mean_dino_side_no_c = lat_mean_dino_scores_side_no_c.mean()

    
    gt_CLIP_mean = torch.tensor(gt_CLIP).mean()
    

    dino_scores = []
    n = len(gt_DINO)
    for i in range(n):
        dino_emb_1 = gt_DINO[i]
        for j in range((i+1), n):
            dino_emb_2 = gt_DINO[j]
            dino_sim = dino_score(dino_emb_1, dino_emb_2)
            dino_scores.append(dino_sim)
            
    facenet_scores = []
    facenet_gt_embeddings = list(map(lambda x:x[0], filter(lambda x: x[1]==embeddings.class_to_idx[prompt], embeddings)))
    n_embeddings = len(facenet_gt_embeddings)
    for i in range(n_embeddings):
        for j in range(i + 1, n_embeddings):
            facenet_sim = (facenet_gt_embeddings[i] - facenet_gt_embeddings[j]).norm().detach().cpu()
            facenet_scores.append(facenet_sim)

    dinos = torch.tensor(dino_scores)
    facenets = torch.tensor(facenet_scores)
    gt_dino_mean = dinos.mean()
    facenet_mean = facenets.mean()

    clip_dino_scores = {
        'Condition': ['Front view with color', 'Side view with color', 'Front view without color', 'Side view without color'],
        'CLIP Score': [CLIP_score_front_c.detach().cpu().numpy(), CLIP_score_side_c.detach().cpu().numpy(), CLIP_score_front_no_c.detach().cpu().numpy(), CLIP_score_side_no_c.detach().cpu().numpy()],
        'DINO Score': [dino_front_c.detach().cpu().numpy(), dino_side_c.detach().cpu().numpy(), dino_front_no_c.detach().cpu().numpy(), dino_side_no_c.detach().cpu().numpy()],
        'lat mean CLIP Score': [lat_mean_CLIP_score_front_c.detach().cpu().numpy(), lat_mean_CLIP_score_side_c.detach().cpu().numpy(), lat_mean_CLIP_score_front_no_c.detach().cpu().numpy(), lat_mean_CLIP_score_side_no_c.detach().cpu().numpy()],
        'lat mean DINO Score': [lat_mean_dino_front_c.detach().cpu().numpy(), lat_mean_dino_side_c.detach().cpu().numpy(), lat_mean_dino_front_no_c.detach().cpu().numpy(), lat_mean_dino_side_no_c.detach().cpu().numpy()]
    }
    
    all_clip_scores = np.array(clip_dino_scores['CLIP Score'])
    all_dino_scores = np.array(clip_dino_scores['DINO Score'])
    
    lat_mean_all_clip_scores = np.array(clip_dino_scores['lat mean CLIP Score'])
    lat_mean_all_dino_scores = np.array(clip_dino_scores['lat mean DINO Score'])

    mean_clip_scores = np.mean(all_clip_scores, axis=0)
    mean_dino_scores = np.mean(all_dino_scores, axis=0)

    lat_mean_mean_clip_scores = np.mean(lat_mean_all_clip_scores, axis=0)
    lat_mean_mean_dino_scores = np.mean(lat_mean_all_dino_scores, axis=0)

    df = pd.DataFrame.from_dict(clip_dino_scores, orient='index')
    
    
    print('###########################################################################')
    print(f'results for {prompt}')
    print(f'CLIP & DINO_')
    print(df)
    
    print("Facenet:")
    print(facenet_all_scores)
    
    lat_mean_facenet_all_scores, lat_mean_facenet_prompt_score, lat_mean_facenet_rest_score
    
    table = {
        "": ["Latent Mean", "Ours", "Ground Truth"],
        "CLIP": [float(lat_mean_mean_clip_scores.squeeze()), float(mean_clip_scores.squeeze()), float(gt_CLIP_mean)],
        "DINO": [lat_mean_mean_dino_scores, mean_dino_scores, float(gt_dino_mean)],
        "FaceNet correct": [lat_mean_facenet_prompt_score, facenet_prompt_score, float(facenet_mean)],
        "FaceNet others": [lat_mean_facenet_rest_score, facenet_rest_score, 0.]
    }
    
    table_df = pd.DataFrame.from_dict(table, orient="index")
    print("quantitive mean table")
    print(table_df)
    
    return {
        "CLIP/DINO": df,
        "All Facenet": facenet_all_scores,
        "table": table_df
    }

#lat_rep = torch.load(lat_rep_path)
#lat_rep = pickle.load(open(lat_rep_path, "rb"))
#lat_rep = [tensor.to(device) for tensor in lat_rep]

#with torch.no_grad():
#    get_scores(lat_rep, prompt, gt_image_paths)
