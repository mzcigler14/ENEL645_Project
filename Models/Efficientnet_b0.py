from Models.ModelBase import BaseModelInterface
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from LearningAndTesting import TrainEvalTest
from torchvision.transforms import InterpolationMode

class efficientnet_b0_model(BaseModelInterface):
  def __init__(self, device, logger, data_handler):
    
    self.device = device
    self.logger = logger
    self.data_handler = data_handler
    self.model = None
    self.trainer_evaluator = None

  def prepare_model(self, learning_rate, batch_size):

    model = torchvision.models.efficientnet_b0(pretrained=True)

    #freeze convolutional layers
    for param in model.parameters():
      param.requires_grad = True
    
    # Customize the fully connected layer
    model.classifier[1] = nn.Linear(in_features = 1280, out_features=7)


    model = model.to(self.device)

    # Define model parameters
    criterion = nn.CrossEntropyLoss()
    # only look at new top parameters (fully connected)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # decrease learning reat by 0.1 every 5 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    self.model = model

    data_loaders, dataset_sizes = self.data_handler.get_data(batch_size, InterpolationMode.BICUBIC, 256, 224)

    self.trainer_evaluator = TrainEvalTest(model, criterion, optimizer, scheduler, data_loaders, dataset_sizes, self.device, self.logger, self.__class__.__name__, learning_rate, batch_size)

  
  def train_model(self, epochs, patience):
    self.trainer_evaluator.train_model(epochs, patience)

  def test_model(self):
    self.trainer_evaluator.test_model()

    
