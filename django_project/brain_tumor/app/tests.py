from django.test import TestCase
from .models import Patient, Scan, AnalyseReport

class PatientModelTest(TestCase):
    def test_create_patient(self):
        patient = Patient.objects.create(
            first_name="Иван",
            last_name="Иванов",
            date_of_birth="1990-01-01",
            gender="Мужской"
        )
        self.assertEqual(patient.first_name, "Иван")