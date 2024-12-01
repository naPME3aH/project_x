from django.urls import path
from django.contrib.auth import views as auth_views
from django.conf.urls.static import static
from django.conf import settings
from . import views

urlpatterns = [
    path(
        'login/',
        auth_views.LoginView.as_view(
            template_name='login.html',  # Ваш шаблон
            redirect_authenticated_user=True,  # Перенаправление, если уже авторизован
        ),
        name='login'
    ),  # Страница авторизации
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),  # Выход
    path('', views.index_view, name='index'),  # Главная страница с перенаправлением
    path('patients/', views.patient_list_view, name='patient_list'),  # Список пациентов (главная после входа)
    path('patients/<int:id>/', views.patient_detail_view, name='patient_detail'),  # Страница пациента
    path('patients/<int:id>/scan/<int:scan_id>/', views.analyze_scan_view, name='analyze_scan'),  # Страница скана
    path('patients/<int:id>/report/<str:report_id>/', views.report_view, name='report_view'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)