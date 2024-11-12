import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Параметры
batch_size = 32
data_dir = 'F:/Project_x/CNN_model/Class_4'
model_path = 'tumor_model.pth'

# Определение нейронной сети
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

# Функция для тестирования модели
def test_model():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = TumorClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    test_model()