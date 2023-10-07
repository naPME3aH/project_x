from django.shortcuts import render
from .forms import UserRegistrationForm
from django.http import HttpResponseRedirect
from django.urls import reverse

def register_user(request):
    form = UserRegistrationForm()

    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)

        if form.is_valid():
            form.save()

            return HttpResponseRedirect(reverse('login'))

    return render(request, 'users/registration.html', {'form':form})

def home(request):
    return render(request,"users/login.html")