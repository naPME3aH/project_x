import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Параметры
model_path = 'tumor_model.pth'
img_path = 'F:/Project_x/CNN_model/Class_4test/normal/N_35.jpg'

# Протестированная выборка
img_path_normal = 'F:/Project_x/CNN_model/Class_4/test/normal/N_18.jpg'                    # Предсказанный class: meningioma_tumor, Confidence: 0.47
img_path_pituitary_tumor = 'F:/Project_x/CNN_model/Class_4/test/pituitary_tumor/P_26.jpg'  # Предcказанный class: pituitary_tumor, Confidence: 0.74
img_path_glioma_tumor = 'F:/Project_x/CNN_model/Class_4/test/glioma_tumor/G_22.jpg'        # Предсказанный class: glioma_tumor, Confidence: 0.55
img_path_meningioma_tumor = 'F:/Project_x/CNN_model/Class_4/test/meningioma_tumor/M_2.jpg' # Предсказанный class: meningioma_tumor, Confidence: 0.86

# Названия классов
class_names = ['glioma_tumor', 'meningioma_tumor', 'normal', 'pituitary_tumor']

# Определение сверточной нейронной сети
class TumorClassifier(nn.Module):
    def __init__(self):
        super(TumorClassifier, self).__init__()
        # Сверточные и пулинг слои
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Определение полносвязных слоев (размер входа нужно пересчитать)
        self.fc1 = nn.Linear(128 * 32 * 32, 128)  # Пересчитанная размерность
        self.fc2 = nn.Linear(128, 4)  # 4 класса для классификации

    def forward(self, x):
        # Применяем сверточные и пулинг-слои
        x = self.pool(torch.relu(self.conv1(x)))  # Сверточный слой 1 с пулингом
        x = self.pool(torch.relu(self.conv2(x)))  # Сверточный слой 2 с пулингом
        x = self.pool(torch.relu(self.conv3(x)))  # Сверточный слой 3 с пулингом
        
        # Преобразуем данные для полносвязных слоев
        x = x.view(x.size(0), -1)  # Разворачивание тензора
        x = torch.relu(self.fc1(x))  # Первый полносвязный слой
        x = self.fc2(x)  # Второй полносвязный слой (выход)
        return x

# Функция для тестирования на конкретном изображении
def test_specific_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    model = TumorClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    image = Image.open(img_path)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    # Вывод имени класса
    class_name = class_names[predicted_class.item()]
    print(f'Предсказанный class: {class_name}, Confidence: {confidence.item():.2f}')

if __name__ == "__main__":
    # test_specific_image(img_path)
    print("img_path_normal")
    test_specific_image(img_path_normal)

    # print("img_path_pituitary_tumor")
    # test_specific_image(img_path_pituitary_tumor)

    # print("img_path_glioma_tumor")
    # test_specific_image(img_path_glioma_tumor)

    # print("img_path_meningioma_tumor")
    # test_specific_image(img_path_meningioma_tumor)
