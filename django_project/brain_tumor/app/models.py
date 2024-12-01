from django.db import models

# from django.db import models

# class Patient(models.Model):
#     first_name = models.CharField(max_length=50)
#     last_name = models.CharField(max_length=50)
#     middle_name = models.CharField(max_length=50, null=True, blank=True)
#     date_of_birth = models.DateField()
#     gender = models.CharField(max_length=10)

#     class Meta:
#         db_table = 'patients'  # Указываем, что модель Patient связана с таблицей patients

#     def __str__(self):
#         return f"{self.first_name} {self.last_name}"


# class Scan(models.Model):
#     patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
#     scan_date = models.DateField()

#     class Meta:
#         db_table = 'scans'  # Указываем, что модель Scan связана с таблицей scans

#     def __str__(self):
#         return f"Scan {self.id} for {self.patient}"


# class Patient(models.Model):
#     first_name = models.CharField(max_length=50)
#     last_name = models.CharField(max_length=50)
#     date_of_birth = models.DateField()
#     gender = models.CharField(max_length=10)

#     def __str__(self):
#         return f"{self.first_name} {self.last_name}"

# class Scan(models.Model):
#     patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
#     scan_date = models.DateField()
#     file_path = models.CharField(max_length=255)

# class AnalyseReport(models.Model):
#     patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
#     scan = models.ForeignKey(Scan, on_delete=models.CASCADE)
#     type_anomaly = models.CharField(max_length=255)
#     report_date = models.DateField(auto_now_add=True)
#     diagnosis = models.TextField()
#     recommendations = models.TextField()