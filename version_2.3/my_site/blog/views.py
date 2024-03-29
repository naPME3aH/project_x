from django.shortcuts import render, get_object_or_404
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.auth.models import User # импортируем модель User
from django.views.generic import (
    ListView,
    DetailView,
    CreateView,
    UpdateView,
    DeleteView
)
from .models import Post


def home(request):
    context = {
        'posts': Post.objects.all()
    }
    return render(request, 'blog/home.html', context)


class PostListView(ListView):
    model = Post
    template_name = 'blog/home.html'  
    context_object_name = 'posts'
    ordering = ['-date_posted']


class PostDetailView(DetailView):
    model = Post


class PostCreateView(LoginRequiredMixin, CreateView):
    model = Post
    fields = ['title', 'content']

    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)


class PostUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Post
    fields = ['title', 'content']

    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)

    def test_func(self):
        post = self.get_object()
        if self.request.user == post.author:
            return True
        return False


class PostDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Post
    success_url = '/'

    def test_func(self):
        post = self.get_object()
        if self.request.user == post.author:
            return True
        return False
    
# def chat(request):
#     return render(request, 'blog/chat_app.html', {'title': 'Чат'})


def about(request):
    return render(request, 'blog/about.html', {'title': 'О клубе Python Bites'})

def user_profile(request, username):
    user = get_object_or_404(User, username=username) # получаем объект пользователя по его нику
    # posts = Post.objects.filter(author=user).order_by('-date') # получаем все посты, написанные этим пользователем, в обратном хронологическом порядке
    context = {'user': user} # создаем словарь с контекстом для шаблона
    return render(request, 'blog/profileuser.html', context) # возвращаем ответ с отрендеренным шаблоном

