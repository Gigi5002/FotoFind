import os
import torch
from torchvision import models, transforms
from PIL import Image
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api.models import Product
from product_search.api.models import Product

def get_image_features(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)
    model = models.resnet50(pretrained=True)
    model.eval()
    with torch.no_grad():
        features = model(image).numpy()
    return features

if __name__ == "__main__":
    data_dir = 'data/images'
    features = {}

    for filename in os.listdir(data_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(data_dir, filename)
            features[filename] = get_image_features(image_path)

    with open('data/features.pkl', 'wb') as f:
        pickle.dump(features, f)
