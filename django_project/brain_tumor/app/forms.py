from django import forms
from django.contrib.auth.forms import AuthenticationForm

class CustomLoginForm(AuthenticationForm):
    username = forms.CharField(max_length=255, label="Логин")
    password = forms.CharField(widget=forms.PasswordInput, label="Пароль")