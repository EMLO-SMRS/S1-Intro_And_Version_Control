import os
import time
import math

import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm



# --------------Training Method-------------------------------------------
def train(model, device, train_loader, criterion, optimizer, epoch):
    """
    trains the neural network once for complete dataset. Adds the Accracy and Loss to the metrics dictionary
    Args: 
        model: neural network
        device: cuda or cpu
        train_loader: Train Dataset Loader
        criterion: Loss Function
        optimizer: Optimiser function
        echo: Number for logging
    """
    train_loss = 0.0
    correct = 0
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _,pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred==target).item()
        pbar.set_description(desc= f'epoch = {epoch} loss={loss.item()} batch_id={batch_idx}')
    metrics['train_accuracy'].append(100 * correct / len(train_loader.dataset))
    metrics['train_loss'].append(train_loss/len(train_loader))

# -------------- Testing Method ------------------------------------------------
def test(model, device, test_loader, criterion):
    """
    tests the neural network once for complete dataset. Adds the total Accracy, Loss, classwise accuracy to the metrics dictionary.
    Args:
        model: neural network
        device: cuda or cpu
        train_loader: Train Dataset Loader
        criterion: Loss Function
        optimizer: Optimiser function
        echo: Number for logging
    """
    model.eval()
    test_loss = 0
    correct = 0
    class_correct = {i: {'correct': 0, 'total': 0}  for i in range(len(class_names))}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            test_loss += criterion(outputs, target).item()  # sum up batch loss
            _,pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred==target).item()
            for i in class_correct:
                class_correct[i]['correct'] += torch.sum((pred == target) * (target == i)).item()
                class_correct[i]['total'] += torch.sum((target == i)).item()

    test_loss /= len(test_loader)
    metrics['test_accuracy'].append(100 * correct / len(test_loader.dataset))
    metrics['test_loss'].append(test_loss)

    # Class Wise Accuracy
    for key, value in class_label_maping.items():
        _acc = class_correct[value]['correct']/class_correct[value]['total'] * 100
        metrics[key].append(_acc)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)')


# -------------- Main Train Model ------------------------------------------------------------

def train_model(model, criterion, optimizer, num_epochs=2, checkpoint=None):
    """
    A main method which calls both train and test method for each epoch and saves the best model
    Args:
        model: neural network
        device: cuda or cpu
        train_loader: Train Dataset Loader
        criterion: Loss Function
        optimizer: Optimiser function
        echo: Number for logging
    """
    since = time.time()
    if checkpoint is None:
        best_loss = math.inf
    else:
        print(f'Val loss: {checkpoint["best_loss"]}')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint['best_loss']

    for epoch in range(num_epochs):
        metrics['epoch'].append(epoch + 1)
        train(model, device, dataloaders['train'], criterion, optimizer, epoch)
        test(model, device, dataloaders['validation'], criterion)
        # Check if the current model has less loss than best model and update accordingly
        if metrics['test_loss'][-1] < best_loss:
            best_loss = metrics['test_loss'][-1]
            torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            }, CHECK_POINT_PATH)

    time_elapsed = time.time() - since
    print(f"Training is Complete: Time taken is {time_elapsed//60}min {time_elapsed%60}sec")

# -------------------- Write Training Summary Markdown file -------------------------------------------------------
def write_summary():
    summary = "<h1 align='center'>Training Summary</h1>\n"
    summary += "\n"
    summary += "## Data\n"
    data_params = [[
        'Total Train Data Set', dataset_sizes["train"]
        ],
        [
            'Total Test Data Set', dataset_sizes['validation']
        ],
        [
            'Total Classes', len(class_names)
        ],
        [
            'Class Names', ", ".join(class_names)
        ],
    ]
    summary += pd.DataFrame(data_params, columns=['Description', 'value']).to_markdown()
    summary += "\n"
    summary += "## Model\n"
    model_params = [[
        'Model ', 'Resnet 18(Pretrained)'
        ],
        [
            'Optimiser', 'SGD'
        ],
        [
            'Loss function ', 'Cross Entropy(lr=0.001, momentum=0.9)'
        ],
        [
            'Class Names', ", ".join(class_names)
        ],
    ]
    summary += pd.DataFrame(model_params, columns=['Description', 'value']).to_markdown()
    summary += "\n"
    summary += "## Training\n"
    summary += metrics_df.to_markdown()
    summary += "\n"

    summary += "## Graphs\n"
    plt.plot(metrics['epoch'], metrics['train_loss'], label = "Train Loss")
    plt.plot(metrics['epoch'], metrics['test_loss'], label = "Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Graph")
    plt.legend()
    plt.savefig('loss.jpg')
    plt.close()
    plt.plot(metrics['epoch'], metrics['train_accuracy'], label = "Train Accuracy")
    plt.plot(metrics['epoch'], metrics['test_accuracy'], label = "Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Graph")
    plt.legend()
    plt.savefig('accuracy.jpg')
    plt.close()
    plt.plot(metrics['epoch'], metrics[class_names[0]], label = f"Class Wise Accuracy of {class_names[0]}")
    plt.plot(metrics['epoch'], metrics[class_names[1]], label = f"Class Wise Accuracy of {class_names[1]}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Class Wise Accuracy Graph")
    plt.legend()
    plt.savefig('class_wise.jpg')

    summary += "![](loss.jpg)"
    summary += "\n"
    summary += "![](accuracy.jpg)"
    summary += "\n"
    summary += "\n"
    summary += "![](class_wise.jpg)"
    summary += "\n"

    with open("Summary.md", "w") as f:
        f.write(summary)

# -------------------- Data Transformation -----------------------------------------------------------------------

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# ----------------------------------------------------------------------
DATA_DIR = 'data'
CHECK_POINT_PATH = 'model.pt'
METRICS_FILE = 'metrics.csv'
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                          data_transforms[x])
                  for x in ['train', 'validation']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=0)
              for x in ['train', 'validation']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}

class_names = image_datasets['train'].classes
class_label_maping = image_datasets['train'].class_to_idx

print(class_names)
print(f'Train image size: {dataset_sizes["train"]}')
print(f'Validation image size: {dataset_sizes["validation"]}')
metrics = {'epoch': [], "train_accuracy": [], "train_loss": [], "test_accuracy": [], "test_loss": []}

# Columns for class wise accuracy
for i in class_names:
    metrics[i] = []

model = torchvision.models.resnet18(pretrained=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# for param in model.parameters():
#     param.requires_grad = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

try:
    checkpoint = torch.load(CHECK_POINT_PATH)
    print("checkpoint loaded")
except Exception:
    checkpoint = None
    print("checkpoint not found")


train_model(model,
            criterion,
            optimizer,
            num_epochs=10,
            checkpoint=checkpoint)

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(METRICS_FILE, index=False)
write_summary()









