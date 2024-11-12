import os
import random
import shutil

# Параметры
data_dir = 'F:/Project_x/CNN_model/Class_4'
train_size = 0.8

# Функция для разделения данных
def split_data():
    classes = ['glioma_tumor', 'meningioma_tumor', 'normal', 'pituitary_tumor']
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        files = os.listdir(class_dir)
        random.shuffle(files)

        train_count = int(len(files) * train_size)
        train_files = files[:train_count]
        test_files = files[train_count:]

        # Создание директорий
        os.makedirs(os.path.join(data_dir, 'train', class_name), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'test', class_name), exist_ok=True)

        for file in train_files:
            shutil.copy(os.path.join(class_dir, file), 
                        os.path.join(data_dir, 'train', class_name, file))
        for file in test_files:
            shutil.copy(os.path.join(class_dir, file), 
                        os.path.join(data_dir, 'test', class_name, file))

if __name__ == "__main__":
    split_data()