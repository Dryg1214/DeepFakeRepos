import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from torchvision.datasets import ImageFolder
import argparse
from matplotlib import pyplot as plt
import torch.nn.functional as F

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Neural Network Training Parameters')
        self.add_arguments()

    def add_arguments(self):
        # Добавление аргументов
        self.parser.add_argument('--device', type=str, default='cuda', help='Device for training: cpu or cuda')
        self.parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
        self.parser.add_argument('--epochs', type=int, default=35, help='Number of training epochs')
        self.parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for SGD optimizer')
        self.parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay coefficient')
        # Добавьте здесь другие параметры, если необходимо
        
    def parse_args(self):
        # Парсинг аргументов командной строки
        return self.parser.parse_args()


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        pretrained_model = models.resnet18(pretrained=True)
        num_features = pretrained_model.fc.in_features
        self.features = nn.Sequential(
            *list(pretrained_model.children())[:-1]
        )
        self.fc_hidden = nn.Linear(num_features, 1024)
        self.fc_output = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc_hidden(x)
        x = torch.relu(x)
        x = self.fc_output(x)
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
    return real_as_real, fake_as_fake


# Инициализация модели
model = ResNet18().to(args.device)  # Предполагаем, что два класса (например, настоящие и фальшивые изображения)
summary(model, input_size=(3, 1024, 1024), device="cpu")
print(model)