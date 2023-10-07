from django.urls import path
from users import views


urlpatterns = [
    
    path('reg/', views.register_user, name='register'),
    path('', views.home, name = "home")

]