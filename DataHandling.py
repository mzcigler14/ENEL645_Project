import os
import yaml
import random
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import numpy as np

class DataHandler:
  def __init__(self, path, crop):
    self.path = path
    self.crop = crop
    self.mean = [0.485, 0.456, 0.406]
    self.std = [0.229, 0.224, 0.225]
    self.labels = [[], [], []]  # List to hold three lists, one for each set
    self.file_paths = [[], [], []]  # List to hold three lists, one for each set
    self.label_idxs = [[], [], []]  # List to hold three lists, one for each set

    image_path = os.path.join(path, crop, 'images')
    metadata_path = os.path.join(path, crop, 'labels_trainval.yml')

    assert os.path.exists(image_path), "{} does not exist, please check your root_path".format(image_path)
    assert os.path.exists(metadata_path), "{} does not exist, please check your root_path".format(metadata_path)

    trainval_file_names, trainval_labels_dict, class_to_ind = DataUtils.get_metadata(metadata_path)
    train_files, val_files, test_files = DataUtils.train_val_test_split(trainval_file_names, 0.7, 0.15)

    # Populate lists for train set
    for file_name in train_files:
        self.labels[0].append(trainval_labels_dict[file_name])
        self.file_paths[0].append(os.path.join(image_path, file_name))
        self.label_idxs[0].append(class_to_ind[trainval_labels_dict[file_name]])

    # Populate lists for validation set
    for file_name in val_files:
        self.labels[1].append(trainval_labels_dict[file_name])
        self.file_paths[1].append(os.path.join(image_path, file_name))
        self.label_idxs[1].append(class_to_ind[trainval_labels_dict[file_name]])

    # Populate lists for test set
    for file_name in test_files:
        self.labels[2].append(trainval_labels_dict[file_name])
        self.file_paths[2].append(os.path.join(image_path, file_name))
        self.label_idxs[2].append(class_to_ind[trainval_labels_dict[file_name]])

    # Convert lists to NumPy arrays
    self.labels = [np.array(sublist) for sublist in self.labels]
    self.file_paths = [np.array(sublist) for sublist in self.file_paths]
    self.label_idxs = [np.array(sublist) for sublist in self.label_idxs]    

   


  def get_data(self, batch_size, interpolation_mode, resolution, crop):

    transform = {
    'Train': transforms.Compose([
    transforms.Resize(resolution, interpolation=interpolation_mode),  # Resize images to 256x256 using bilinear interpolation
    transforms.CenterCrop(crop),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(180),
    transforms.ToTensor(),  # Convert images to Tensor
    transforms.Normalize(self.mean, self.std)  # CHANGE THESE TO THE TRAINF MEAN ANS STD
    ]),

    'Validation': transforms.Compose([
    transforms.Resize(resolution, interpolation=interpolation_mode),  # Resize images to 256x256 using bilinear interpolation
    transforms.CenterCrop(crop),
    transforms.ToTensor(),  # Convert images to Tensor
    transforms.Normalize(self.mean, self.std)  # CHANGE THESE TO THE TRAINF MEAN ANS STD
    ]),

    'Test': transforms.Compose([
    transforms.Resize(resolution, interpolation=interpolation_mode),  # Resize images to 256x256 using bilinear interpolation
    transforms.CenterCrop(crop),
    transforms.ToTensor(),  # Convert images to Tensor
    transforms.Normalize(self.mean, self.std)  # CHANGE THESE TO THE TRAINF MEAN ANS STD
  ])}

    self.datasets = {
      'Train': CustomDataset(self.file_paths[0], self.label_idxs[0], transform['Train']),
      'Validation': CustomDataset(self.file_paths[1], self.label_idxs[1], transform['Validation']),
      'Test': CustomDataset(self.file_paths[2], self.label_idxs[2], transform['Test'])
    } 

    data_loaders = {}
    for x in self.datasets.keys():
        if x is not None:
            data_loaders[x] = DataLoader(
                self.datasets[x],
                batch_size=batch_size,
                shuffle=(x =="Train"),
                num_workers=2
            )

    dataset_sizes = {x: len(self.datasets[x]) for x in ['Train', 'Validation', 'Test']}

    return data_loaders, dataset_sizes
  


class CustomDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class DataUtils:
  @staticmethod
  def train_val_test_split(data, train_proportion, val_proportion):
    total_size = len(data)
    train_size = int(total_size*train_proportion)
    val_size = int(total_size*val_proportion)

    random.shuffle(data)
    train_set = data[:train_size]
    val_set = data[train_size: train_size + val_size]
    test_set = data[train_size + val_size: ]

    return train_set, val_set, test_set

  @staticmethod
  def get_metadata(path):

    class_to_ind = {
      'unfertilized': 0,
      '_PKCa': 1,
      'N_KCa': 2,
      'NP_Ca': 3,
      'NPK_': 4,
      'NPKCa': 5,
      'NPKCa+m+s': 6,
    }

    # load metadata
    labels_trainval = yaml.safe_load(open(path, 'r')) # dict, e.g., {20200422_1.jpg: unfertilized, ...}
    file_names = list(labels_trainval.keys())

    return file_names, labels_trainval, class_to_ind

