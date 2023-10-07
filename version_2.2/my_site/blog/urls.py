from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='blog-home'),
    path('chat/', views.chat, name='blog-chat'),
    path('about/', views.about, name='blog-about'),
]