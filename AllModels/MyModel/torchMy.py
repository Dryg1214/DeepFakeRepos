import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
#from networks.AIDetection import AIDetectionCNN
from torchvision.datasets import ImageFolder
import argparse

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Neural Network Training Parameters')
        self.add_arguments()

    def add_arguments(self):
        # Добавление аргументов
        self.parser.add_argument('--device', type=str, default='cpu', help='Device for training: cpu or cuda')
        self.parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
        self.parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
        self.parser.add_argument('--train_data_path', type=str, default='C:\\DatasetD\\TORCH\\GHOST_100', help='Path to training data')
        self.parser.add_argument('--val_data_path', type=str, default='C:\\DatasetD\\TORCH\\ValidationGHOST', help='Path to validation data')
        self.parser.add_argument('--test_data_path', type=str, default='D:\\DeepFakeRepos\\DatasetsUnique\\GhostData', help='Path to test data')
        self.parser.add_argument('--model_save_path', type=str, default='C:\\DatasetD\\TORCH\\', help='Directory to save trained models')
        self.parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for SGD optimizer')
        self.parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay coefficient')
        # Добавьте здесь другие параметры, если необходимо
        
    def parse_args(self):
        # Парсинг аргументов командной строки
        return self.parser.parse_args()
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG_Like(nn.Module):
    def __init__(self, input_shape, num_classes=2, num_filters=32):
        super(VGG_Like, self).__init__()

        # Первый блок свертки
        self.conv1 = nn.Conv2d(input_shape[0], num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Второй блок свертки
        self.conv4 = nn.Conv2d(num_filters, num_filters*2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(num_filters*2, num_filters*2, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(num_filters*2, num_filters*2, kernel_size=3, padding=1)

        # Третий блок свертки
        self.conv7 = nn.Conv2d(num_filters*2, num_filters*4, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(num_filters*4, num_filters*4, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(num_filters*4, num_filters*4, kernel_size=3, padding=1)

        # Четвертый блок свертки
        self.conv10 = nn.Conv2d(num_filters*4, num_filters*8, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(num_filters*8, num_filters*8, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(num_filters*8, num_filters*8, kernel_size=3, padding=1)

        # Пятый блок свертки
        self.conv13 = nn.Conv2d(num_filters*8, num_filters*16, kernel_size=3, padding=1)
        self.conv14 = nn.Conv2d(num_filters*16, num_filters*16, kernel_size=3, padding=1)
        self.conv15 = nn.Conv2d(num_filters*16, num_filters*16, kernel_size=3, padding=1)

        # Финальный слой перед GlobalMaxPooling
        self.conv16 = nn.Conv2d(num_filters*16, num_filters*16, kernel_size=5, padding=2)

        # GlobalMaxPooling
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))

        # Выходной слой
        self.fc = nn.Linear(num_filters*16, num_classes)

    def forward(self, x):
        # Первый блок свертки
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # Второй блок свертки
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)

        # Третий блок свертки
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.pool(x)

        # Четвертый блок свертки
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.pool(x)

        # Пятый блок свертки
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        x = F.relu(self.conv15(x))
        x = self.pool(x)

        # Финальный слой перед GlobalMaxPooling
        x = F.relu(self.conv16(x))

        # GlobalMaxPooling
        x = self.global_pool(x)

        # Выходной слой
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        
        return x
    


arg_parser = ArgumentParser()
args = arg_parser.parse_args()

def culc_confusion_matrix(train_labels, binary_predictions):
    cm = confusion_matrix(train_labels, binary_predictions)
    #print("Confusion Matrix  (Test Set):")
    #print(cm)

    # Вычисление точности каждой категории на тестовой выборке
    real_as_real = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    real_as_fake = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    fake_as_fake = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    fake_as_real = cm[1, 0] / (cm[1, 0] + cm[1, 1])

    print(f"Accuracy (Real as Real): {100 * real_as_real:.2f}%, "
          f"Accuracy (Real as Fake): {100 * real_as_fake:.2f}%, "
          f"Accuracy (Fake as Fake): {100 * fake_as_fake:.2f}%, "
          f"Accuracy (Fake as Real): {100 * fake_as_real:.2f}%")


train_file = 'train_dataset_CNN.pt'
test_file = 'test_dataset_CNN.pt'

if os.path.exists(train_file):
    train_dataset = torch.load(train_file)
else:
    train_dataset = ImageFolder(root=args.train_data_path, transform=transforms.ToTensor())
    torch.save(train_dataset, train_file)
if os.path.exists(test_file):
    test_dataset = torch.load(test_file)
else:
    test_dataset = ImageFolder(root=args.test_data_path, transform=transforms.ToTensor())
    torch.save(test_dataset, test_file)

# Предполагаем, что у вас есть загруженные данные и созданы DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Инициализация модели
model = VGG_Like(input_shape=(3, 1024, 1024), num_classes=2).to(args.device)  # Предполагаем, что два класса (например, настоящие и фальшивые изображения)
#criterion = nn.CrossEntropyLoss().to(args.device)
criterion = nn.BCEWithLogitsLoss().to(args.device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
model.train()
# Обучение модели
for epoch in range(args.epochs):
    running_loss = 0.0
    train_labels = []
    train_predictions = []
    print(f"Epoch [{epoch+1}/{args.epochs}]")
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(args.device), labels.to(args.device)

        optimizer.zero_grad()
        outputs = model(inputs)
        one_hot_labels = nn.functional.one_hot(labels, 2)
        loss = criterion(outputs, one_hot_labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

        outputs = torch.sigmoid(outputs)  # Преобразуем выходы в бинарные предсказания
        _, preds = torch.max(outputs, 1)
        train_labels.extend(labels.cpu().numpy())
        train_predictions.extend(preds.detach().cpu().numpy())
    epoch_loss = running_loss / len(train_loader.dataset)
    #print(f"Loss: {epoch_loss / len(train_loader)}")
    culc_confusion_matrix(train_labels, train_predictions)

    torch.save(model, "" + f'modelAIDetection_epoch_{epoch+1}.pth')

    print("Test data acc")
    # Вычисление матрицы ошибок на обучающей выборке
    model.eval()
    train_preds = []
    train_targets = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)  # Преобразуем выходы в бинарные предсказания
            _, preds = torch.max(outputs, 1)
            train_labels.extend(labels.cpu().numpy())
            train_predictions.extend(preds.detach().cpu().numpy())
    culc_confusion_matrix(train_labels, train_predictions)