import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Параметры
model_path = 'F:/Project_x/brain_tumor/app/cnn/tumor_model.pth'

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

    try:
        image = Image.open(img_path).convert('RGB')  # Открываем и конвертируем изображение
        image = transform(image).unsqueeze(0)  # Применяем трансформации и добавляем batch dimension

        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        # Получаем имя класса и уверенность
        class_name = class_names[predicted_class.item()]
        
        # Возвращаем предсказанный класс и уверенность
        return class_name, confidence.item()
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        return None, None

if __name__ == "__main__":
    # Примеры использования
    # Подставьте путь к изображению для тестирования
    img_path_example = 'F:/Project_x/scans/1/11.jpg'
    predicted_class, confidence_score = test_specific_image(img_path_example)
    print(f"Предсказанный класс: {predicted_class}, Уверенность: {confidence_score}")
