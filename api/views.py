import os
import torch
from torchvision import models, transforms
from PIL import Image
from django.shortcuts import render
from django.http import JsonResponse
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from .models import Product

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

def search_similar_products(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        image_features = get_image_features(image)

        with open('data/features.pkl', 'rb') as f:
            features_db = pickle.load(f)

        similarities = {}
        for path, features in features_db.items():
            similarity = cosine_similarity(image_features, features.reshape(1, -1))[0][0]
            similarities[path] = similarity

        sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        top_similarities = sorted_similarities[:5]

        results = []
        for path, similarity in top_similarities:
            product = Product.objects.get(image=path)
            results.append({
                'name': product.name,
                'description': product.description,
                'image': product.image.url,
                'similarity': similarity,
            })

        if not results:
            return JsonResponse({'results': 'К сожалению, похожих товаров не найдено.'})
        else:
            return JsonResponse({'results': results})
    else:
        return JsonResponse({'error': 'No image provided'})

def index(request):
    return render(request, 'index.html')
