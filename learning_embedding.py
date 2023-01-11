

!pip install timm
import numpy as np 
import pandas as pd
from PIL import Image
import pickle
import timm
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
import os
import gc
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from torch import tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

device

def move_to(obj):
  if isinstance(obj, dict):
    res = {}
    for k, v in obj.items():
      res[k] = move_to(v)
    return res
  elif isinstance(obj, np.ndarray):
    return torch.from_numpy(obj)
  else:
    raise TypeError("Invalid type for move_to")

class MyModel3(nn.Module):
  def __init__(self):
    super().__init__()
    model = timm.create_model('swin_large_patch4_window7_224' , pretrained  = True  , num_classes = 0)
    self.layer = nn.AdaptiveAvgPool1d(64)
    self.feature_extractor = model
  def forward(self, x):
    x = transforms.functional.resize(x,size=[224, 224])
    x = x/255.0
    x = transforms.functional.normalize(x, 
                                            mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])
    x = self.layer(self.feature_extractor(x))
    return x
model = MyModel3()
model.eval()
saved_model = torch.jit.script(model)
saved_model.save('saved_model.pt')

class MyModel2(nn.Module):
  def __init__(self):
    super().__init__()
    model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
    model.eval()
    model.head = nn.AdaptiveAvgPool1d(64)
    self.feature_extractor = model
  def forward(self, x):
    x = transforms.functional.resize(x,size=[224, 224])
    x = x/255.0
    x = transforms.functional.normalize(x, 
                                            mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])
    x = self.feature_extractor(x)
    
    return x
model = MyModel2()
model.eval()
saved_model = torch.jit.script(model)
saved_model.save('saved_model.pt')

model = MyModel2()
model

for name, param in model.named_parameters():
    print(name)

class MyModel(nn.Module):
  def __init__(self):
    super().__init__()
    model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
    model.eval()
    model.head = nn.Linear(768,64)
    self.feature_extractor = model
  def forward(self, x):
    x = transforms.functional.resize(x,size=[224, 224])
    x = x/255.0
    x = transforms.functional.normalize(x, 
                                            mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])
    x = self.feature_extractor(x)
    
    return x
# model = MyModel()
# model.eval()
# saved_model = torch.jit.script(model)
# saved_model.save('saved_model.pt')

!conda create -n rapids-22.06 -c rapidai -c nvidia -c conda-forge rapids=22.06 python=3.9 cudatoolkit=11.5 -y

dataset = pd.read_csv('../input/triplet-data/datas.csv')

dataset = pd.read_csv('../input/images-for-google-comp/embdatakaggoo.csv')

dataset = cudf.read_csv('../input/images-for-google-comp/embdatakaggoo.csv')

dataset.device

dicr = {"galary":2871,"toys":3408,"dishes":12919,"house":17563,"landmarks": 25830}

for index,i in enumerate(dataset['inputs']):
    if index<2872:
        listw.append((i,dataset.iloc[0:2871,0].sample().iloc[0],dataset.iloc[2872:,0].sample().iloc[0]))
    elif index<3409:
        listw.append((i,dataset.iloc[2872:3408,0].sample().iloc[0],pd.concat([dataset.iloc[0:2871,0] ,dataset.iloc[3409:,0]]).sample().iloc[0]))
    elif index<12920:
        listw.append((i,dataset.iloc[3409:12919,0].sample().iloc[0],pd.concat([dataset.iloc[0:3408,0] ,dataset.iloc[12920:,0]]).sample().iloc[0]))
    elif index<17564:
        listw.append((i,dataset.iloc[12920:17563,0].sample().iloc[0],pd.concat([dataset.iloc[0:12919,0] ,dataset.iloc[17564:,0]]).sample().iloc[0]))
    elif index<=25830:
        listw.append((i,dataset.iloc[17564:25830,0].sample().iloc[0],dataset.iloc[0:17563,0].sample().iloc[0]))

datasa = dataset.sample(frac=1).reset_index(drop=True)

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file):
        self.data = annotations_file

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input = self.data.iloc[idx, 0]
        if idx<2872:
            pos = dataset.iloc[0:2871,0].sample().iloc[0]
            neg = dataset.iloc[2872:,0].sample().iloc[0]
        elif idx<3409:
            pos = dataset.iloc[2872:3408,0].sample().iloc[0]
            neg = pd.concat([dataset.iloc[0:2871,0] ,dataset.iloc[3409:,0]]).sample().iloc[0]
        elif idx<12920:
            pos = dataset.iloc[3409:12919,0].sample().iloc[0]
            neg = pd.concat([dataset.iloc[0:3408,0] ,dataset.iloc[12920:,0]]).sample().iloc[0]
        elif idx<17564:
            pos = dataset.iloc[12920:17563,0].sample().iloc[0]
            neg = pd.concat([dataset.iloc[0:12919,0] ,dataset.iloc[17564:,0]]).sample().iloc[0]
        elif idx<=25830:
            pos = dataset.iloc[17564:25830,0].sample().iloc[0]
            neg = dataset.iloc[0:17563,0].sample().iloc[0]
        input_image = Image.open(input).convert("RGB")
        input_tensor = convert_to_tensor(input_image)
        input_batch = input_tensor.unsqueeze(0)
        input_batch_anc = transforms.functional.resize(input_batch[0],size=[224, 224])
        input = pos
        input_image = Image.open(input).convert("RGB")
        input_tensor = convert_to_tensor(input_image)
        input_batch = input_tensor.unsqueeze(0)
        input_batch_pos = transforms.functional.resize(input_batch[0],size=[224, 224])
        input = neg
        input_image = Image.open(input).convert("RGB")
        input_tensor = convert_to_tensor(input_image)
        input_batch = input_tensor.unsqueeze(0)
        input_batch_neg = transforms.functional.resize(input_batch[0],size=[224, 224])
        return input_batch_anc,input_batch_pos,input_batch_neg

training_data = CustomImageDataset(dataset)

def custom_euc_dis(y_predictions, target):
  loss_value = torch.square(y_predictions - target).sum().sqrt()
  return loss_value

def metric(y_predictions, target):
    wanted_value = torch.square(y_predictions - target).sum(0).sqrt()
    return torch.mean(wanted_value)

class TripletLoss(nn.Module):
    def __init__(self, margin=0.9):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()

k_folds = 8
  num_epochs = 10
  loss_function = torch.jit.script(TripletLoss())
  
  # For fold results
  results = {}
  
  # Set fixed random number seed
  torch.manual_seed(42)
  
  # Define the K-fold Cross Validator
  kfold = KFold(n_splits=k_folds, shuffle=True)
    
  # Start print
  print('--------------------------------')

  # K-fold Cross Validation model evaluation
  for fold, (train_ids, test_ids) in enumerate(kfold.split(training_data)):
    
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
                      training_data, 
                      batch_size=512, sampler=train_subsampler,num_workers = 2,pin_memory=True)
    testloader = torch.utils.data.DataLoader(
                      training_data,
                      batch_size=512, sampler=test_subsampler,num_workers = 2,pin_memory=True)
    
    # Init the neural network
    network = MyModel2()
    network.to(device)
    for name, param in network.named_parameters():
        if param.requires_grad and not '11.mlp' in name:
            param.requires_grad = False
    # Initialize optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=16*1e-4)
    
    # Run the training loop for defined number of epochs
    for epoch in range(0, num_epochs):

      # Print epoch
      print(f'Starting epoch {epoch+1}')

      # Set current loss value
      running_loss = []

      # Iterate over the DataLoader for training data
      for i, (anc,pos,neg) in enumerate(trainloader, 0):
        
        # Get inputs
        anc,pos,neg = anc.to(device),pos.to(device),neg.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        out_anc = network(anc)
        out_pos = network(pos)
        out_neg = network(neg)
        # Compute loss
        loss = loss_function(out_anc,out_pos,out_neg)
        
        # Perform backward pass
        loss.backward()
        
        # Perform optimization
        optimizer.step()
        
        # Print statistics
        running_loss.append(loss.detach())
      print("Epoch: {} - Loss: {:.4f}".format(epoch+1, torch.mean(torch.tensor(running_loss))))
    # Process is complete.
    print('Training process has finished. Saving trained model.')

    # Print about testing
    print('Starting testing')
    
    # Saving the model
    save_path = f'./model-fold-{fold}.pth'
    torch.save(network.state_dict(), save_path)

    # Evaluationfor this fold
    correct_p,correct_n,total = [],[],0
    with torch.no_grad():

      # Iterate over the test data and generate predictions
      for i, (anc,pos,neg) in enumerate(testloader, 0):

        # Get inputs
        anc,pos,neg = anc.to(device),pos.to(device),neg.to(device)

        # Generate output
        out_anc = network(anc)
        out_pos = network(pos)
        out_neg = network(neg)

        # Set total and correct
        total+=1024
        predicted_p = metric(out_anc,out_pos)
        predicted_n = metric(out_anc,out_neg)
        correct_p.append(predicted_p)
        correct_n.append(predicted_n)
      print(torch.mean(torch.tensor(correct_p)),torch.mean(torch.tensor(correct_n)),torch.mean(torch.tensor(correct_p))<torch.mean(torch.tensor(correct_n)))
      print(total)
      print('--------------------------------')
      break

save_path = f'./model-fold-{fold}.pth'
torch.save(network.state_dict(), save_path)

gc.collect()

!wget https://s3.amazonaws.com/google-landmark/train/images_000.tar

model = MyModel()

import tarfile
my_tar = tarfile.open('/kaggle/working/images_000.tar')
my_tar.extractall('./images') # specify which folder to extract to
my_tar.close()

convert_to_tensor = transforms.Compose([transforms.PILToTensor()])

directory ="../input/house-rooms-image-dataset/House_Room_Dataset/Bedroom"
for filename in os.listdir(directory):
    input_image = Image.open(os.path.join(directory, filename)).convert("RGB")
    input_tensor = convert_to_tensor(input_image)
    input_batch = input_tensor.unsqueeze(0)
    print(input_batch.shape)
    with torch.no_grad():
        k = model(input_batch)[0]
        print(k.shape)
        break
        mean+=k
        count+=1
    del input_image
    gc.collect()

!ls

from zipfile import ZipFile

with ZipFile('submission.zip','w') as zip:           
  zip.write('saved_model.pt', arcname='saved_model.pt')

model = MyModel()
model.eval()
saved_model = torch.jit.script(model)
saved_model.save('saved_model.pt')

model = MyModel2()
model.load_state_dict(torch.load('./model-fold-0.pth'))
model.eval()
saved_model = torch.jit.script(model)
saved_model.save('saved_model.pt')

!ls

!rm ./saved_model.pt

