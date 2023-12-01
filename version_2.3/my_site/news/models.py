from django.db import models

class News(models.Model):
    title = models.CharField(max_length=100) # заголовок поста
    content = models.TextField() # содержание поста
    date = models.DateTimeField(auto_now_add=True) # дата и время создания поста

    def __str__(self):
        return self.title