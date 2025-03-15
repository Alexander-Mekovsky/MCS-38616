#Importing necessary libraries
import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torch import nn, optim
from sklearn.metrics import f1_score

#Setting up computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#Loads image path and labels
class ImageDataset(Dataset):
    def __init__(self, img_path, labels, img_dir, transform=None):
        self.img_path = img_path
        self.labels = labels
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            label = self.labels[idx]
            return image, label #For train images
        else:
            return image, img_name #For test images

#Creates a Convolutional Neural Network class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        #4 convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        #Batch normalization for each layer
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        #Flattening and connecting layers
        dummy_input = torch.randn(1, 3, 50, 50)
        x = self.pool(F.relu(self.conv1(dummy_input)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        self.flattened_size = x.numel() // x.shape[0]
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.5)

    #Applies RELU activation, pooling, and batch normalization
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        #print(x.shape) #debugging purposes
        x = torch.flatten(x, start_dim=1)
        #print(x.numel() // x.shape[0]) #debugging purposes
        #print(x.shape) #debugging purposes
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#Loads pretrained model if one exists
def load_pretrained_model(model, path_to_model_weights):
    model.load_state_dict(torch.load(path_to_model_weights))
    model.to(device)
    return model

print('Importing and processing images...')

#Loads training images and processes them using transforms, resizing, nromalization, and conversion to tensor, then splits the data
labeled_data = pd.read_csv('dataset/train_data.csv')
img_paths = labeled_data['img_name']
labels = labeled_data['label']
transform = transforms.Compose([transforms.Resize((50, 50)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = ImageDataset(img_paths, labels, img_dir = 'dataset/train_images', transform = transform)
train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=64, shuffle=True)

model = CNN().to(device)

total_accuracy, total_f1, train_loss, val_loss = [], [], [], []

#Loads pretrained model, if not starts training new model
pretrained_model_path = 'trained_model_4.pth'
if os.path.exists(pretrained_model_path):
    print('Loading pretrained model...')
    model = load_pretrained_model(model, pretrained_model_path)
else:
    print('Pretrained model not found. Proceeding with training new model...')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    #Trains and evaluates model and computes losses for both phases and prints all metrics
    for epoch in range(100):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        train_loss.append(avg_train_loss)
        model.eval()
        with torch.no_grad():
            y_true, y_pred = [], []
            correct = 0
            total = 0
            running_loss = 0.0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                running_loss += loss.item()
        accuracy = 100 * correct / total
        f1 = f1_score(y_true, y_pred, average = 'weighted')
        avg_val_loss = running_loss / len(val_loader)
        print(f'Epoch {epoch + 1}, Accuracy: {accuracy:.2f}%, F1 Score: {f1:.4f}, Train Loss: {avg_train_loss:.4f}, Validate loss: {avg_val_loss:.4f}')
        total_accuracy.append(accuracy)
        total_f1.append(f1)
        val_loss.append(avg_val_loss)

    #Saves model
    torch.save(model.state_dict(), 'trained_model_4.pth')

#Loads and prepares testing images
print('Testing model...')
test_images_loc = 'dataset/test_images'
test_images_path = os.listdir(test_images_loc)
test_dataset = ImageDataset(img_path = test_images_path, labels = None, img_dir = test_images_loc, transform = transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

model.eval()
predictions_list = []

#Predicts which images are parasitized or not
with torch.no_grad():
    for inputs, img_names in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        for i in range(len(predicted)):
            img_name = img_names[i]
            predicted_class = predicted[i].item()
            predictions_list.append([img_name, predicted_class])

#Saves predictions to dataframe and saves it as a .csv file
predictions_df = pd.DataFrame(predictions_list, columns=['img_name', 'label'])
predictions_df.to_csv('predictions.csv', index=False)
print('Predictions saved...')

#Saves accuracy scores and saves it to a dataframe then a .csv file
scores_df = pd.DataFrame({'Accuracy': total_accuracy, 'F1': total_f1, 'Train_Loss': train_loss, 'Val_Loss': val_loss})
scores_df.to_csv('scores.csv', index=False)
print('Scores saved...')

print('End of program.')

