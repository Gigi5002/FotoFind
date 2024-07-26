from django.urls import path
from .views import search_similar_products, index

urlpatterns = [
    path('', index),
    path('search/', search_similar_products),
]
