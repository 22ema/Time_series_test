import os
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, TensorDataset


class DNNModel(nn.Module):
    def __init__(self, inputs):
        super(DNNModel, self).__init__()
        input_layer = nn.Linear(inputs, 256)
        hidden1 = nn.Linear(256, 128)
        hidden2 = nn.Linear(128, 64)
        hidden3 = nn.Linear(64, 32)
        output_layer = nn.Linear(32,16)
        self.hidden = nn.Sequential(
            input_layer,
            nn.ReLU(),
            hidden1,
            nn.ReLU(),
            hidden2,
            nn.ReLU(),
            hidden3,
            nn.ReLU(),
            output_layer,
        )
        if torch.cuda.is_available():
            self.hidden = self.hidden.cuda()

    def forward(self, x):
        o = self.hidden(x)
        return o

def create_dataset(dataset_dir):
  X, y = [], []
  labels = os.listdir(dataset_dir)
  for label in labels:
    file_list = os.listdir(dataset_dir + label + '/')
    for f in file_list:
      temp = pd.read_csv(dataset_dir + label + '/' + f)
      X.append(torch.from_numpy(temp.values))
      y.append(label)
  X = pad_sequence(X, batch_first=True)
  return X, y

def train_model(model, epoch_num, dataloader_dict, criterion, optimizer):
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    for epoch in range(0, epoch_num):
        print("Epoch{}/{}".format(epoch+1, epoch_num))
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)
            epoch_loss = 0.0
            epoch_corrects = 0
            for inputs, labels in tqdm(dataloader_dict[phase]):
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                print(outputs)
                print(labels)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                epoch_loss += loss.item() * inputs.size(0)
                epoch_corrects += torch.sum(preds == labels.data)
            total_epoch_loss = epoch_loss/len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects.double()/len(dataloader_dict[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, total_epoch_loss, epoch_acc
            ))
    return model

if __name__=="__main__":
    batch_size = 256
    lr = 0.0001
    epochs = 10
    input_size = 6
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dataset0_train_path = 'dataset/dataset0/train/'
    dataset0_val_path = 'dataset/dataset0/test/'
    dataset0_classes = os.listdir(dataset0_train_path)

    dataset1_train_path = 'dataset/dataset1/train/'
    dataset1_test_path = 'dataset/dataset1/test/'

    dataset0_train_X, dataset0_train_y = create_dataset(dataset0_train_path)
    dataset0_val_X, dataset0_val_y = create_dataset(dataset0_val_path)
    dataset0_label_encoder = LabelEncoder()
    dataset0_label_encoder.fit(dataset0_classes)
    dataset0_train_y = dataset0_label_encoder.transform(dataset0_train_y)
    dataset0_val_y = dataset0_label_encoder.transform(dataset0_val_y)

    dataset0_train_dataset = TensorDataset(torch.tensor(dataset0_train_X).float(), torch.from_numpy(dataset0_train_y))
    dataset0_val_dataset = TensorDataset(torch.tensor(dataset0_val_X).float(), torch.from_numpy(dataset0_val_y))

    dataset0_train_dataloader = DataLoader(dataset0_train_dataset, batch_size=batch_size)
    dataset0_val_dataloader = DataLoader(dataset0_val_dataset, batch_size=batch_size)
    dataset0_dataloader_dict = {'train' : dataset0_train_dataloader, 'val' : dataset0_val_dataloader}
    criterion = torch.nn.MSELoss()

    model = DNNModel(input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    result_modle = train_model(model, epochs, dataset0_dataloader_dict, criterion, optimizer)