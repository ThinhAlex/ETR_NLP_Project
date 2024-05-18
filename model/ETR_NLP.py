import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR 
import torchvision.models as models

from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader

import parallelTestModule
import multiprocessing

# Shuffle function
def channel_shuffle(x, groups):
  batch_size, num_channels, height, width = x.size()
  assert num_channels % groups == 0

  channels_per_group = num_channels // groups
  x = x.view(batch_size, groups, channels_per_group, height, width)
  x = x.permute(0, 2, 1, 3, 4)
  x = x.reshape(batch_size, num_channels, height, width)
  return x

# C_in, C_out: number of input/output channels
# gamma: sharing ratio of ETR
# prims: settings of NLPs
# T: No. of tasks
class NLP(nn.Module):
  def __init__(self, C_in, C_out, prims):
    super(NLP, self).__init__()
    # define NLPs
    self.ops = nn.ModuleList(prims)
    self.k = len(prims) # No. of NLPs
    # group-wise linear combination
    self.conv1x1 = nn.Conv2d(C_in*self.k, C_out, kernel_size = 1, groups = C_in)

  def forward(self, x):
    # extract features by NLPs
    y = torch.cat([op(x) for op in self.ops], dim = 1)
    # re-arrange features via channel shuffling
    y = channel_shuffle(y, groups = self.k)
    return self.conv1x1(y)

class ETR_NLP(nn.Module):
  def __init__(self, C_in, C_out, gamma, prims, T):
    super(ETR_NLP, self).__init__()
    self.C_shared = int(gamma*C_out)
    C_specif = int(C_out - self.C_shared)
    # define a shared branch
    self.shared_branch = NLP(C_in, C_out, prims)
    # define task-specific branches
    for i in range(T):
      specif_branch = nn.Conv2d(C_in, C_specif*3, kernel_size = 3, stride = 1, padding = 1) # standard Conv
      self.add_module("task_{}".format(i), specif_branch)
    self.task = 0 # set an active task

  def get_layer(self, name):
    return getattr(self, name)

  def update_task(self):
    self.task += 1

  def reset_task(self):
    self.task = 0

  def forward(self, x):
    self.shared = self.shared_branch(x)
    self.specif = self.get_layer("task_{}".format(self.task))(x)
    return torch.cat([self.shared, self.specif], dim = 1)
  
# Average Pooling 2D & Upsampling
class avg_pool(nn.Module):
  def __init__(self, kernel_size):
    super(avg_pool, self).__init__()
    self.kernel_size = kernel_size
    
  def forward(self, input_tensor):
    pooled_image = nn.functional.avg_pool2d(input_tensor, kernel_size=self.kernel_size)
    upsampled_image = nn.functional.interpolate(pooled_image, scale_factor=self.kernel_size, mode='nearest')
    return avg_pool
  
# Shift function
class shift_image(nn.Module):
  def __init__(self, shift):
    super(shift_image, self).__init__()
    self.shift = shift
    
  def forward(self, input_tensor):
      shifted_tensor = torch.roll(input_tensor, shifts=(self.shift, self.shift), dims=(0, 1))  
      return shifted_tensor



# Perturbation function
class Perturbation(nn.Module):
        def __init__(self, mean=0, std=1):
            super(Perturbation, self).__init__()
            self.mean = mean
            self.std = std

        def forward(self, x):
            noise = torch.randn_like(x) * self.std + self.mean
            return x + noise

def main():
  if torch.cuda.is_available():
      device = torch.device("cuda")  # Use GPU
      print("GPU is available.")
  else:
      device = torch.device("cpu")   # Use CPU
      print("GPU is not available, using CPU.")

  ### II. Dataset
    
  root_img = "ETR_NLP/dataset/data_faces/img_align_celeba"
  img_list = os.listdir(root_img)
  print("Number of images:", len(img_list))

  root_labels = "ETR_NLP/dataset/list_attr_celeba.csv"
  df = pd.read_csv(root_labels, encoding='utf-8')
  df = df.drop(columns = ["image_id"])

  # Change -1 to 0
  header = df.columns.tolist()
  df[header] = df[header].map(lambda x: 0 if x == -1 else x)

  # Vectorize 40 attributes
  labels = torch.tensor(df.values)

  # Split data into train, val, test sets
  torch.manual_seed(1)

  celeb_data = datasets.ImageFolder('ETR_NLP/dataset/data_faces', transform=transforms.Compose([
    transforms.Resize((128,128)), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), transforms.transforms.RandomAffine(degrees=15),
transforms.transforms.RandomHorizontalFlip(),
transforms.transforms.ColorJitter(brightness=0.2, contrast=0.2)]))

  train_ratio = 0.7
  test_ratio = 0.15
  val_ratio = 0.15

  num_data = len(celeb_data)
  train_len = int(train_ratio*num_data)
  val_len = int(val_ratio*num_data)
  test_len = num_data - train_len - val_len

  train_celeb, val_celeb, test_celeb = random_split(celeb_data, [train_len, val_len, test_len])
  id = train_celeb.indices # id of shuffled images

  # DataLoader
  celeb_batch = 256

  train_loader_celeb = DataLoader(
      train_celeb,
      batch_size = celeb_batch,
      shuffle = True,
      num_workers = 6,
      drop_last = True,
      pin_memory = True,
  )

  val_loader_celeb = DataLoader(
      val_celeb,
      batch_size = celeb_batch,
      shuffle = False,
  )

  test_loader_celeb = DataLoader(
      test_celeb,
      batch_size = celeb_batch,
      shuffle = False,
  )     
      
  # Training    
  torch.manual_seed(100)
  T = 40
  C_in = 3
  C_out = 3
  gamma = 0.8
  prims = [
      shift_image(5),
      shift_image(10),
      nn.Conv2d(in_channels=3, out_channels = 3, kernel_size=3, stride = 1, padding = 1, bias = False),
      Perturbation(mean=0, std=0.1),
      Perturbation(mean=0.5, std=0.2)
  ]
  
  etr_nlp = ETR_NLP(C_in, C_out, gamma, prims, T)
  resnet = models.resnet18(pretrained = True)
  
  in_fts = resnet.fc.in_features
  resnet.fc = nn.Linear(in_fts, 2)
  
  optimizer = Adam(resnet.parameters(), lr = 0.0001)
  scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
  
  resnet.to(device)
  etr_nlp.to(device)

  epochs = 10
  train_loss = []
  val_loss = []
  train_accuracy = []
  val_accuracy = [] 
  try:
    for epoch in range(epochs):
      resnet.train()
      etr_nlp.train()
      start = 0
      end = celeb_batch

      total_train_loss = 0
      total_val_loss = 0
      total_true = 0
      for idx, (X_train, _) in enumerate(train_loader_celeb):
        X_train = X_train.to(device)
        etr_nlp.reset_task()
        for i in range(T):
          layers = etr_nlp(X_train) # get torch.cat([shared, specif], dim = 1)

          # Plug each branch to resnet model
          shared_branch = resnet(etr_nlp.shared)
          specif_branch = resnet(etr_nlp.specif)

          # Get labels
          y_train = torch.tensor([labels[j,i] for j in id[start:end]]).to(device)

          # Loss each branch
          loss_shared = nn.CrossEntropyLoss()(shared_branch, y_train)
          loss_specif = nn.CrossEntropyLoss()(specif_branch, y_train)

          # Losses both branches
          total_loss = loss_shared + loss_specif
          total_train_loss += loss_shared.item() + loss_specif.item()

          # Update
          optimizer.zero_grad()
          total_loss.backward()
          optimizer.step()

          pred = torch.argmax(shared_branch.data, 1)
          compare = pred == y_train
          total_true += torch.sum(compare)

          etr_nlp.update_task()

        start += celeb_batch
        end += celeb_batch

        # Average loss
        total_train_loss = total_train_loss/T

      # Train accuracy
      train_acc_gain = total_true/(celeb_batch*len(train_loader_celeb)*T)
      train_accuracy.append(train_acc_gain)

      # Train loss
      total_train_loss = total_train_loss/(idx+1)
      train_loss.append(total_train_loss)

      print('\nEpoch: {}/{}, Train Loss: {:.4f}, Train Acc: {:.4f}'.format(epoch, epochs, total_train_loss, train_acc_gain))
      scheduler.step()
      
    plt.plot(train_loss, label = "Training Loss")
    plt.plot(train_accuracy, label = "Train Accuracy")
    plt.legend()
    plt.show()
      
  except KeyboardInterrupt:
    print("\nKeyboardInterrupt: Saving models...")

    # Save the model when KeyboardInterrupt occurs
    torch.save(resnet.state_dict(), "ETR_NLP/saved_model_scp/resnet.pth")
    torch.save(etr_nlp.state_dict(), "ETR_NLP/saved_model_scp/etr_nlp.pth")
    print("Model saved!")

if __name__ == '__main__':   
  multiprocessing.freeze_support()
  extractor = parallelTestModule.ParallelExtractor()
  extractor.runInParallel(numProcesses=4, numThreads=16)
  main()