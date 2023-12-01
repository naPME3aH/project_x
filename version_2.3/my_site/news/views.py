from django.shortcuts import render
from .models import News

def news(request):
    updates = News.objects.all().order_by('-date') # получаем все посты с обновлениями в обратном хронологическом порядке
    context = {'updates': updates} # создаем словарь с контекстом для шаблона
    return render(request, 'blog/news.html', context) # возвращаем ответ с отрендеренным шаблоном