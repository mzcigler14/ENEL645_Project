from Models.ModelBase import BaseModelInterface
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from LearningAndTesting import TrainEvalTest
import time
import csv
import copy
from sklearn.metrics import confusion_matrix
import torch
#global average pooling layer to make the model smaller
class CNNModel(nn.Module):
    def __init__(self, device, logger, data_handler):
        super().__init__()
        #implement batch normalization
        self.device = device
        self.logger = logger
        self.data_handler = data_handler
        self.model = None
        self.trainer_evaluator = None
        #896x896
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32,kernel_size= (5, 5), stride = 1, padding = 2)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
    
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64,kernel_size= (5, 5), stride = 1, padding = 2)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size = (2,2))
        self.drop2 = nn.Dropout(0.3)
        #448x448


        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128,kernel_size= (5, 5), stride = 1, padding = 2)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size = (2,2))
        self.drop3 = nn.Dropout(0.3)
        #224x224


        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 256,kernel_size= (5, 5), stride = 1, padding = 2)
        self.act4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size = (2,2))
        #112x112
        self.flat = nn.Flatten()

        #256x112x112
        self.fc5 = nn.Linear(in_features= 256*61*61,out_features= 512)
        self.act5 = nn.ReLU()
        self.drop5 = nn.Dropout(0.5)

        self.fc6 = nn.Linear(in_features=512, out_features=7)
    
    def forward(self, x):
        # Forward pass
        x = self.act1(self.conv1(x))
        # x = self.drop1(x)
        x = self.act2(self.conv2(x))
        x = self.pool2(x)
        # x = self.drop2(x)
        x = self.act3(self.conv3(x))
        x = self.pool3(x)
        # x = self.drop3(x)
        x = self.act4(self.conv4(x))
        x = self.pool4(x)
        x = self.flat(x)
        x = self.act5(self.fc5(x))
        # x = self.drop5(x)
        x = self.fc6(x)
        return x




    def prepare_model(self, learning_rate, batch_size):
        self.model_name = "CNN from Scratch"
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.log_csv_filepath = '/home/matjaz.cigler1/Project/Logs/trainValTest_log.csv'
        self.confusion_matrix_csv = '/home/matjaz.cigler1/Project/Logs/confusion_matrix_log.csv'

        model = CNNModel(self.device, self.logger, self.data_handler)
        model = model.to(self.device)

        # Define model parameters
        self.criterion = nn.CrossEntropyLoss()
        # only look at new top parameters (fully connected)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # decrease learning reat by 0.1 every 5 epochs
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

        self.model = model

        self.data_loaders, self.dataset_sizes = self.data_handler.get_data(batch_size)

    def train_model(self, epochs, patience):
        starting_time = time.time()
        min_loss = 1000
        max_accuracy = 0
        best_model = copy.deepcopy(self.model.state_dict())
        epochs_not_improved = 0
        train_loss_list = []
        val_loss_list = []
        train_acc_list = []
        val_acc_list = []


        for epoch in range(epochs):
            self.logger.info('Starting Epoch {}/{}'.format(epoch+1, epochs))
            # logger.info('\n')
            for phase in ['Train', 'Validation']:
                if phase == 'Train':
                    self.model.train()
                else:
                    self.model.eval()
                culm_loss = 0 # total culmulative loss
                culm_correct = 0
                    #train over each image in the training set
                for inputs, labels in self.data_loaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)
                    if phase == 'Train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    culm_loss += loss.item() * inputs.size(0)
                    culm_correct += torch.sum((predictions == labels).float().sum())     
                    if phase == 'Train':
                        self.scheduler.step()
                
                epoch_loss = culm_loss / self.dataset_sizes[phase]
                if phase == 'Train':
                    train_loss_list.append(epoch_loss)
                else:
                    val_loss_list.append(epoch_loss)

                epoch_acc = culm_correct / self.dataset_sizes[phase]
                if phase == 'Train':
                    train_acc_list.append(epoch_acc)
                else:
                    val_acc_list.append(epoch_acc)

                completion_log = '{}, Loss: {:.4f} Accuracy: {:.4f}'.format(phase, epoch_loss, epoch_acc)
                if phase == 'Validation':
                    completion_log += '\n'    
            
                self.logger.info(completion_log)
            
                if phase == 'Validation':
                    if epoch_loss < min_loss:
                        min_loss = epoch_loss
                        max_accuracy = epoch_acc
                        # best_model = copy.deepcopy(self.model.state_dict())
                        epochs_not_improved = 0
                    else:
                        epochs_not_improved += 1
                    
                #log the details to csv
                with open(self.log_csv_filepath, mode='a', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([self.model_name, self.batch_size, self.learning_rate, phase, epoch+1, epoch_loss, epoch_acc.item(), epochs_not_improved])

                #track number of epochs without accuracy change if not improving, stop the model
                            
                    
                if(epochs_not_improved > patience):
                    self.logger.info("EARLY STOPPING\n")
                    time_elapsed = time.time() - starting_time
                    self.logger.info('Training complete in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
                    # self.logger.info('Best Validation Acc: {:4f}'.format(max_accuracy))
                    self.logger.info('Best Validation Loss: {:4f}\n'.format(min_loss))
                    self.logger.info('Best Model Validation Acc: {:4f}\n'.format(max_accuracy))
                    self.model.load_state_dict(best_model)
                    return self.model
                    # self.logger.info('Loss per epoch (Training): {}'.format(train_loss_list))
                    # self.logger.info('Accuracy per epoch(Training): {}'.format(train_acc_list))
                    # self.logger.info('Loss per epoch (Validation): {}'.format(val_loss_list))
                    # self.logger.info('Accuracy per epoch(Validation): {}'.format(val_acc_list))
                    # load best model weights        
            

        time_elapsed = time.time() - starting_time
        self.logger.info('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        self.logger.info('Best Validation Acc: {:4f}\n'.format(max_accuracy))
        # self.logger.info('Loss per epoch (Training): {}'.format(train_loss_list))
        # self.logger.info('Accuracy per epoch(Training): {}'.format(train_acc_list))
        # self.logger.info('Loss per epoch (Validation): {}'.format(val_loss_list))
        # self.logger.info('Accuracy per epoch(Validation): {}'.format(val_acc_list))

        # load best model weights
        self.model.load_state_dict(best_model)
        return self.model
    
    def test_model(self):
        culm_loss = 0 # total culmulative loss
        culm_correct = 0
        all_predictions = []
        all_labels = []
        
        for inputs, labels in self.data_loaders['Test']:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            culm_loss += loss.item() * inputs.size(0)
            culm_correct += torch.sum((torch.argmax(outputs, 1) == labels).float().sum())

        all_predictions.extend(torch.argmax(outputs, 1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        test_loss = culm_loss / self.dataset_sizes['Test']
        test_acc = culm_correct / self.dataset_sizes['Test']

        # self.logger.info(all_labels)
        cm = confusion_matrix(all_labels, all_predictions)
        self.logger.info('Test Case Loss: {:.4f} Accuracy: {:.4f}\n'.format(
                test_loss, test_acc))
        #Put the confusion matrix into a csv
        with open(self.confusion_matrix_csv, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([self.model_name, self.batch_size, self.learning_rate, test_acc.item(), test_loss, cm.flatten()])

        # self.logger.info('Confusion Matrix:')
        # for row in cm:
        #     self.logger.info(' '.join([str(elem) for elem in row]))
        # self.logger.info(cm)
        # self.logger.info('\n')


    
