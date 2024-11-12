import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Гиперпараметры
batch_size = 32
learning_rate = 0.001
num_epochs = 20
data_dir = 'F:/Project_x/CNN_model/Class_4'

# Определение устройства (GPU, если доступен, иначе CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

# Функция для вычисления точности
def accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    return (predicted == labels).sum().item() / labels.size(0)

# Функция для обучения модели
def train_model():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Оставляем размер 256x256
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = TumorClassifier().to(device)  # Переместить модель на устройство
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()  # Установить режим обучения
        running_loss = 0.0
        running_accuracy = 0.0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Переместить данные на устройство
            optimizer.zero_grad()

            # Прямой проход
            outputs = model(images)

            # Вычисление потерь и обратный проход
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Обновление метрик
            running_loss += loss.item()
            running_accuracy += accuracy(outputs, labels) * labels.size(0)
            total_samples += labels.size(0)

        # Вычисление средней потери и точности за эпоху
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = running_accuracy / total_samples

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

    # Сохранение модели
    torch.save(model.state_dict(), 'tumor_model.pth')

if __name__ == "__main__":
    train_model()
