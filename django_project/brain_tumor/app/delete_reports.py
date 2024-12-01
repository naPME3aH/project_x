from pymongo import MongoClient

# Настройка MongoDB
client = MongoClient('mongodb://medmongo:med2mongo4@192.168.1.2:9000')
db = client['medical_records']
anomalies_collection = db['anomalies']

def delete_reports_for_patient(patient_id):
    # Удаление всех отчётов для указанного patient_id
    result = anomalies_collection.delete_many({"patient_id": patient_id})
    print(f"Удалено отчётов: {result.deleted_count} для пациента с ID {patient_id}")

if __name__ == "__main__":
    delete_reports_for_patient(patient_id=1)
