from pymongo import MongoClient
from django.conf import settings
from bson import ObjectId
from .cnn.report_model import generate_report
from datetime import datetime

# Подключение к MongoDB с использованием указанных базы данных и коллекции
client = MongoClient("mongodb://medmongo:med2mongo4@192.168.1.2:9000")
db = client['medical_records']  # Подключаемся к базе данных medical_records
anomalies_collection = db['anomalies']  # Подключаемся к коллекции anomalies

# Функция для сохранения отчета
def save_report(patient_id, scan_id, report_date, type_anomaly, diagnosis, recommendations):
    # Установка значения по умолчанию для diagnosis, если None
    if diagnosis is None:
        diagnosis = "Диагноз не установлен"
    
    report = {
        "patient_id": patient_id,
        "scan_id": scan_id,
        "report_date": report_date,  # Добавляем report_date
        "type_anomaly": type_anomaly,
        "diagnosis": diagnosis,
        "recommendations": recommendations,
    }
    result = anomalies_collection.insert_one(report)
    return result.inserted_id

# Функция для получения отчета
def get_report(report_id=None, patient_id=None, scan_id=None):
    if report_id:
        if not isinstance(report_id, ObjectId):
            report_id = ObjectId(report_id)  # Преобразование строки в ObjectId
        return anomalies_collection.find_one({"_id": report_id})
    elif patient_id and scan_id:
        # Используем patient_id и scan_id для поиска отчета
        return anomalies_collection.find_one({"patient_id": patient_id, "scan_id": scan_id})
    else:
        return None