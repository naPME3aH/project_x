from django.shortcuts import render
from .models import Post


def home(request):
    context = {
        'posts': Post.objects.all()
    }
    return render(request, 'blog/home.html', context)

def chat(request):
    return render(request, 'blog/chat_app.html', {'title': 'Чат'})

def about(request):
    return render(request, 'blog/about.html', {'title': 'О клубе Melissa'})