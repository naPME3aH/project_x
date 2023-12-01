from django.urls import path
from . import views

urlpatterns = [
    path('', views.news, name='blog-news'), # URL-адрес для списка постов с обновлениями
]