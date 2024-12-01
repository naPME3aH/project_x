from django.conf import settings
from django.db import connection
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login
from .mongodb import save_report, get_report
from .cnn.specific_test import test_specific_image
from .cnn.report_model import generate_report
import os
from datetime import datetime

def index_view(request):
    if request.user.is_authenticated:
        return redirect('patient_list')
    return redirect('login')

def login_view(request):
    error_message = None  # Переменная для хранения сообщения об ошибке
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)  # Проверяем учетные данные

        if user is not None:
            login(request, user)  # Авторизуем пользователя
            return redirect('patient_list')  # Перенаправляем на страницу списка пациентов
        else:
            error_message = "Неверный логин или пароль"  # Устанавливаем сообщение об ошибке

    # Возвращаем ту же страницу с возможным сообщением об ошибке
    return render(request, 'login.html', {'error_message': error_message})

@login_required
def patient_list_view(request):
    with connection.cursor() as cursor:
        cursor.execute("SELECT patient_id, first_name, last_name, midle_name, date_of_birth FROM patients ORDER BY patient_id ASC")
        patients = cursor.fetchall()

    patients_list = [
        {
            "id": row[0],
            "first_name": row[1],
            "last_name": row[2],
            "middle_name": row[3] if row[3] is not None else "",
            "date_of_birth": row[4]
        }
        for row in patients
    ]

    return render(request, 'patient_list.html', {'patients': patients_list})

@login_required
def patient_detail_view(request, id):
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT patient_id, first_name, last_name, midle_name, date_of_birth, gender FROM patients WHERE patient_id = %s", [id]
        )
        patient_row = cursor.fetchone()

    if not patient_row:
        return render(request, '404.html')

    patient = {
        "id": patient_row[0],
        "first_name": patient_row[1],
        "last_name": patient_row[2],
        "middle_name": patient_row[3] if patient_row[3] is not None else "",
        "date_of_birth": patient_row[4],
        "gender": patient_row[5]
    }

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT scan_id, scan_date 
            FROM scans 
            WHERE patient_id = %s 
            ORDER BY scan_date DESC
            """, [id]
        )
        scans = cursor.fetchall()

    scans_list = [
        {
            "id": scan[0],
            "date": scan[1],
            "image_url": f"/media/{id}/{id}{scan[0]}.jpg"
        }
        for scan in scans
    ]

    return render(request, 'patient_detail.html', {'patient': patient, 'scans': scans_list})


@login_required
def analyze_scan_view(request, id, scan_id):
    # Извлечение данных пациента по ID из PostgreSQL
    with connection.cursor() as cursor:
        cursor.execute("SELECT first_name, last_name, midle_name FROM patients WHERE patient_id = %s", [id])
        patient_row = cursor.fetchone()

    # Если пациент не найден
    if not patient_row:
        return render(request, '404.html', {'message': 'Пациент с данным ID не найден'}, status=404)

    # Формируем ФИО пациента
    patient_name = f"{patient_row[1]} {patient_row[0]} {patient_row[2] if patient_row[2] else ''}".strip()

    # Путь к файлу изображения для анализа
    media_root = os.path.join(settings.BASE_DIR, "app/media")  # Корневая директория медиафайлов
    img_path = os.path.join(media_root, str(id), f"{id}{scan_id}.jpg")  # Формируем полный путь

    # Проверяем, существует ли файл изображения
    if not os.path.exists(img_path):
        print("Файл изображения не найден:", img_path)
        return render(request, '404.html', {'message': 'Файл изображения не найден'}, status=404)

    # Проверяем, существует ли уже отчет для данного скана
    existing_report = get_report(patient_id=id, scan_id=scan_id)
    if existing_report:
        print(f"Отчет уже существует в MongoDB: {existing_report}")
        # Если отчет найден, перенаправляем на страницу просмотра отчета
        return redirect('report_view', id=id, report_id=str(existing_report['_id']))

    # Если отчет не найден, выполняем анализ изображения
    try:
        anomaly_type, confidence = test_specific_image(img_path)  # Анализ изображения
    except FileNotFoundError:
        print("Ошибка чтения файла:", img_path)
        return render(request, '404.html', {'message': 'Ошибка чтения файла'}, status=404)

    # Генерация текста отчета с использованием модели NLP
    report_content = generate_report(
        patient_name=patient_name,  # Передаем ФИО пациента
        examination_date=datetime.now().strftime('%d.%m.%Y'),
        preliminary_diagnosis=anomaly_type
    )

    # Сохранение отчета в MongoDB
    report_id = save_report(
        patient_id=id,
        scan_id=scan_id,
        type_anomaly=anomaly_type,
        diagnosis=anomaly_type,
        recommendations=report_content,
        report_date=datetime.now()
    )

    print(f"Отчет успешно сохранен в MongoDB с ID: {report_id}")

    # Перенаправляем на страницу отчета после создания
    return redirect('report_view', id=id, report_id=str(report_id))


@login_required
def report_view(request, id, report_id):
    print(f"Received report_id: {report_id} (type: {type(report_id)})")
    # Получение отчета из MongoDB по ID или возврат 404, если не найден
    report = get_report(report_id=report_id)
    if not report:
        return render(request, '404.html', status=404)  # Показываем 404, если отчет не найден
    
    # Получение ФИО пациента из PostgreSQL
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT first_name, last_name, midle_name
            FROM patients
            WHERE patient_id = %s
        """, [id])
        patient_row = cursor.fetchone()
    
    if not patient_row:
        return render(request, '404.html', status=404)

    # Формирование полного имени пациента
    patient_full_name = f"{patient_row[1]} {patient_row[0]} {patient_row[2]}" if patient_row[2] else f"{patient_row[1]} {patient_row[0]}"

    # Словарь для перевода диагнозов
    diagnosis_translation = {
        "glioma_tumor": "Глиома",
        "meningioma_tumor": "Менингиома",
        "pituitary_tumor": "Опухоль гипофиза",
        "normal": "Нет опухоли"
    }

    # Перевод диагноза
    diagnosis = diagnosis_translation.get(report['diagnosis'], report['diagnosis'])

    # Передача данных в шаблон
    context = {
        'report': report,
        'patient_full_name': patient_full_name,
        'scan_number': report['scan_id'],  # Здесь вы можете преобразовать scan_id в "Номер снимка"
        'translated_diagnosis': diagnosis  # Переведенный диагноз
    }
    return render(request, 'report.html', context)


