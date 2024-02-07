import torch
import clip

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

def calculate_clip(img):
    image = preprocess(img[0]).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    print(image_features.shape)
    return image_features.cpu().numpy()

