import os
import sys
import django
from django.conf import settings
from django.db import connection

# Добавляем корневой каталог проекта в sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Устанавливаем настройки Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "brain_tumor.settings")
django.setup()

def update_scans_table():
    # Путь к папке с медиафайлами
    media_root = os.path.join(settings.BASE_DIR, "app/media")

    # Проверяем, существует ли папка
    if not os.path.exists(media_root):
        print(f"Папка {media_root} не существует.")
        return

    # Идем по каждому подкаталогу (например, "1", "2" - это ID пациента)
    for patient_id_folder in os.listdir(media_root):
        patient_folder_path = os.path.join(media_root, patient_id_folder)
        
        # Проверяем, что это папка и имя состоит только из цифр
        if os.path.isdir(patient_folder_path) and patient_id_folder.isdigit():
            patient_id = int(patient_id_folder)

            # Ищем все файлы с расширением .jpg в этой папке
            for filename in os.listdir(patient_folder_path):
                if filename.endswith(".jpg"):
                    # Имя файла должно быть в формате "{id}{scan_id}.jpg"
                    file_id_part = filename.replace(".jpg", "")

                    # Убеждаемся, что имя файла начинается с patient_id
                    if file_id_part.startswith(patient_id_folder):
                        # Извлекаем scan_id (остаток после patient_id)
                        scan_id = int(file_id_part[len(patient_id_folder):])

                        # Проверяем, есть ли запись с таким patient_id и scan_id в базе данных
                        with connection.cursor() as cursor:
                            cursor.execute(
                                """
                                SELECT COUNT(*) FROM scans
                                WHERE patient_id = %s AND scan_id = %s
                                """,
                                [patient_id, scan_id]
                            )
                            exists = cursor.fetchone()[0]

                        # Если записи нет, добавляем ее
                        if not exists:
                            with connection.cursor() as cursor:
                                cursor.execute(
                                    """
                                    INSERT INTO scans (patient_id, scan_id, scan_date)
                                    VALUES (%s, %s, NOW())  -- Используйте текущую дату
                                    """,
                                    [patient_id, scan_id]
                                )
                                print(f"Добавлено: patient_id={patient_id}, scan_id={scan_id}, scan_date=NOW()")
                        else:
                            print(f"Запись уже существует: patient_id={patient_id}, scan_id={scan_id}")

if __name__ == "__main__":
    update_scans_table()
