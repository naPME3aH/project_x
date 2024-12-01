import requests
import json
from googletrans import Translator

# Данные пациента, которые могут изменяться
# patient_name = "Ivanov Ivan Ivanovich"
# examination_date = "07.11.2024"
# preliminary_diagnosis = "glioma_tumor"  # Эта переменная может изменяться

def generate_report(patient_name, examination_date, preliminary_diagnosis):
    # Динамический шаблон с использованием f-strings
    input_text = f"""
    You are a medical assistant specializing in generating professional HTML reports for patients. Strictly follow the given format and content. 

    Do not add any extra comments, explanations, or text beyond what is requested in this template. 
    Strictly use the HTML structure and tags provided. Do not use markdown syntax like '**' or stray from the specified format.
    Instead of (), write the text in one sentence what is {preliminary_diagnosis} disease.
    Also, (if applicable) specifies a field that may not be applicable for {preliminary_diagnosis}, you must either leave it or remove it.
    Dictionary for translation: {preliminary_diagnosis}. "glioma_tumor": "Glioma", "meningioma_tumor": "Meningioma", "pituitary_tumor": "Pituitary".

    <p>The Patient has been diagnosed with <strong>{preliminary_diagnosis}</strong>. () </p>

    <h3>Recommendations:</h3>
    <ul>
        <li><strong>Neurosurgeon:</strong> To assess the need and possibility of surgical intervention.</li>
        <li><strong>Oncologist:</strong> To develop an individual treatment plan, taking into account the type and stage of the tumor.</li>
        <li><strong>Radiologist:</strong> For planning and possible implementation of radiation therapy.</li>
        <li><strong>Psychologist:</strong> To provide emotional and psychological support to the patient.</li>
    </ul>

    <h3>Additional research:</h3>
    <ul>
        <li>MRI with contrast for accurate tumor boundary and size analysis.</li>
        <li>CT scan to evaluate any effect on adjacent bone structures.</li>
        <li>Biopsy for histological analysis to confirm the tumor type.</li>
        <li>PET scan (if applicable) to evaluate tumor metabolic activity.</li>
    </ul>

    <h3>Treatment Options:</h3>
    <ul>
        <li><strong>Surgical Removal:</strong> A primary treatment option to reduce tumor size and relieve pressure on surrounding brain areas.</li>
        <li><strong>Radiation Therapy:</strong> To address remaining tumor cells post-surgery or as a primary approach if surgery is not feasible.</li>
        <li><strong>Chemotherapy:</strong> Recommended based on malignancy level, especially for {preliminary_diagnosis}.</li>
        <li><strong>Hormonal Therapy:</strong> (if applicable) To correct any hormone imbalances.</li>
    </ul>

    <h3>Disclaimer:</h3>
    <p>This report is a preliminary assessment based on available data. A detailed evaluation and personalized treatment approach should be developed in consultation with a healthcare provider.</p>
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

        return full_text
        # Обработка перевода, игнорируя None элементы
        try:
            translated_text = translator.translate(full_text, src='en', dest='ru').text
            if translated_text:
                print("Сгенерированный отчет и рекомендации (на русском):")
                # print(translated_text)
                return translated_text
            else:
                print("Ошибка: перевод текста не был выполнен.")
        except Exception as e:
            print("Ошибка при переводе текста:", e)
    else:
        print(f"Ошибка: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    patient_name = "Иванов Иван Иванович"
    examination_date = "23.11.2024"
    preliminary_diagnosis = "meningioma_tumor"
    report = generate_report(patient_name, examination_date, preliminary_diagnosis)
    print(report)