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
from models import MyModel,MyModel2,MyModel3
from data_preprocess import preprocess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    Dataset = preprocess()
    training_data = CustomImageDataset(dataset)

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


if __main__ == '__main__':
    main()
