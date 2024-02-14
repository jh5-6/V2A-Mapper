import torch
import clip

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# def calculate_clip(img):
#     image = preprocess(img[0]).unsqueeze(0).to(device)
#     with torch.no_grad():
#         image_features = model.encode_image(image)
#     print(image_features.shape)
#     return image_features.cpu().numpy()

def calculate_clip(images):
    processed_images = torch.stack([preprocess(img).unsqueeze(0).to(device) for img in images])
    with torch.no_grad():
        image_features = [model.encode_image(image) for image in processed_images]
    average_features = torch.mean(torch.stack(image_features), dim=0)
    return average_features.cpu().numpy()

