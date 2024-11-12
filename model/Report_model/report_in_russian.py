import requests
import json
from googletrans import Translator

# Данные пациента, которые могут изменяться
patient_name = "Ivanov Ivan Ivanovich"
examination_date = "07.11.2024"
preliminary_diagnosis = "glioma_tumor"  # Эта переменная может изменяться

# Динамический шаблон с использованием f-strings
input_text = f"""
Patient Report

Patient Name: {patient_name}
Date of Examination: {examination_date}
Preliminary Diagnosis: {preliminary_diagnosis}

---

### Report Summary:
Based on the analysis conducted on {examination_date}, a {preliminary_diagnosis} has been identified. 

#### Detailed Description:
Provide a description based on the characteristics of a {preliminary_diagnosis}.

---

### Recommendations:

#### Specialist Consultations:
1. **Neurosurgeon** - To assess the necessity and feasibility of surgical intervention.
2. **Oncologist** - To devise a personalized treatment plan considering the type and stage of the tumor.
3. **Radiologist** - For planning and potential implementation of radiation therapy.
4. **Psychologist** - To provide emotional and psychological support to the patient.

#### Additional Examinations:
- MRI with contrast for precise tumor boundary and size analysis.
- CT scan to evaluate any effect on adjacent bone structures.
- Biopsy for histological analysis to confirm the tumor type.
- PET scan (if applicable) to evaluate tumor metabolic activity.

#### Treatment Options:
- **Surgical Removal**: Primary treatment option to reduce tumor size and relieve pressure on surrounding brain areas.
- **Radiation Therapy**: To address remaining tumor cells post-surgery or as a primary approach if surgery is not feasible.
- **Chemotherapy**: Recommended based on malignancy level, especially for {preliminary_diagnosis}.
- **Hormonal Therapy** (if applicable) - To correct any hormone imbalances.

*Note: Regular follow-ups and monitoring are recommended based on the patient's response to treatment.*

---

### Disclaimer:
This report is a preliminary assessment based on available data. A detailed evaluation and personalized treatment approach should be developed in consultation with a healthcare provider.

---

Please address this report with appropriate medical oversight, and refer the patient for immediate follow-up with the recommended specialists.
"""

# Настройка данных для запроса
payload = {
    "model": "llama3.2:3b-instruct-q8_0",
    "prompt": input_text,
    "max_tokens": 200
}

# Отправка запроса и обработка ответа
response = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)
translator = Translator()

if response.status_code == 200:
    full_text = ""
    for line in response.iter_lines():
        if line:
            line_json = json.loads(line.decode('utf-8'))
            full_text += line_json.get("response", "")
    
    # Перевод текста на русский
    translated_text = translator.translate(full_text, src='en', dest='ru').text
    print("Сгенерированный отчет и рекомендации (на русском):")
    print(translated_text)
else:
    print(f"Ошибка: {response.status_code}")
    print(response.text)
