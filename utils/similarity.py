import torch
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, InterpolationMode
import clip
from transformers import AutoImageProcessor, AutoModel
device = "cuda" if torch.cuda.is_available() else "cpu"
# --- CLIP ---
CLIP_model, CLIP_preprocess = clip.load("ViT-B/32", device=device)
# --- DINO ---
DINO_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
DINO_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)


def CLIP_similarity(image, gt_embedding):

    # --- CLIP Preprocessing --- 
    clip_tensor_preprocessor = Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=None),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    image_c_first = image.permute(2, 0, 1)
    with torch.no_grad():
        image_preprocessed = clip_tensor_preprocessor(image_c_first).unsqueeze(0)
        CLIP_embedding = CLIP_model.encode_image(image_preprocessed) # [1, 512]
        CLIP_embedding /= CLIP_embedding.norm(dim=-1, keepdim=True)

        CLIP_similarity = 100 * torch.matmul(CLIP_embedding, gt_embedding.T)
    
    return CLIP_similarity


def DINO_similarity(image, gt_embedding):
    with torch.no_grad():
        input = DINO_processor(images=image, return_tensors="pt", do_rescale=False).to(device)
        output = DINO_model(**input)
        CLS_token = output.last_hidden_state # [1, 257, 768]
        DINO_embedding  = CLS_token.mean(dim=1) # [1, 768]
        DINO_embedding /= DINO_embedding.norm(dim=-1, keepdim=True)

        DINO_similarity = 100 * torch.matmul(DINO_embedding, gt_embedding.T)
    
    return DINO_similarity