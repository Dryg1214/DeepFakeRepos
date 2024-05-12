import torch
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from sklearn.metrics import roc_auc_score

dataFolders = []
# data_test_path1 = "D:\\DeepFakeRepos\\DatasetsUnique\\GhostData"
# data_test_path2 = "D:\\DeepFakeRepos\\DatasetsUnique\\RoopData"
# data_test_path3 = "D:\\DeepFakeRepos\\DatasetsUnique\\OlderData"
# data_test_path4 = "D:\\DeepFakeRepos\\DatasetsUnique\\SmileData"
# dataFolders.append(data_test_path1)
# dataFolders.append(data_test_path2)
# dataFolders.append(data_test_path3)
# dataFolders.append(data_test_path4)
batch_size = 4

data_test_path = "D:\\DeepFakeRepos\\DatasetsUnique\\FaceForensics"
dataFolders.append(data_test_path)



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
    
def culc_confusion_matrix(train_labels, binary_predictions):
    cm = confusion_matrix(train_labels, binary_predictions)

    real_as_real = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    real_as_fake = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    fake_as_fake = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    fake_as_real = cm[1, 0] / (cm[1, 0] + cm[1, 1])

    print(f"Accuracy (Real as Real): {100 * real_as_real:.2f}%, "
        f"Accuracy (Real as Fake): {100 * real_as_fake:.2f}%, "
        f"Accuracy (Fake as Fake): {100 * fake_as_fake:.2f}%, "
        f"Accuracy (Fake as Real): {100 * fake_as_real:.2f}%")
    
    # Вычисление ROC AUC
    roc_auc = roc_auc_score(train_labels, binary_predictions)
    print(f"ROC AUC: {roc_auc:.4f}")
    
#model_path = 'D:\DeepFakeRepos\AllModels\MyModel\modelDetect_Resnet18_Custom_GHOST_epoch_35_rr0.9975031210986267_ff0.9962546816479401_1_1.pth'
#model_path = 'D:\DeepFakeRepos\AllModels\MyModel\modelDetect_Resnet18_Custom_ROOP_epoch_34_rr1.0_ff1.0_1_2.pth'
#model_path = 'D:\DeepFakeRepos\AllModels\MyModel\modelDetect_Resnet18_Custom_OLDER_epoch_35_rr0.9975_ff0.99625_1_2.pth'
model_path = 'D:\DeepFakeRepos\AllModels\MyModel\modelDetect_Resnet18_Custom_SMILE_epoch_29_rr1.0_ff1.0_1_2.pth'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path, map_location=device)
model.eval()

desired_size = (1024, 720)
transform = transforms.Compose([
    transforms.Resize(desired_size), 
    transforms.ToTensor()
])

for test_data_path in dataFolders:
    print(test_data_path)
    test_dataset = ImageFolder(root=test_data_path, transform= transform)#transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(test_dataset)
    test_labels = []
    test_predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs) 
            _, preds = torch.max(outputs, 1)
            test_labels.extend(labels.cpu().numpy())
            test_predictions.extend(preds.cpu().numpy())
    culc_confusion_matrix(test_labels, test_predictions)
