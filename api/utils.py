import os
import pickle
import torch
from torchvision import models, transforms
from PIL import Image
from scipy.spatial.distance import cosine

model = models.resnet50(pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image).numpy()
    return features

def get_image_features(image_file):
    image = Image.open(image_file)
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image).numpy()
    return features


def find_similar_images(features_db, query_features, top_k=5):
    similarities = []
    for img_name, db_features in features_db.items():
        sim = cosine(query_features.flatten(), db_features.flatten())
        similarities.append((img_name, sim))
    similarities.sort(key=lambda x: x[1])

    # Получите информацию о найденных продуктах
    top_images = [img_name for img_name, _ in similarities[:top_k]]
    return top_images
