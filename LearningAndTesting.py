import torch
import time
import copy
from sklearn.metrics import confusion_matrix
import csv

class TrainEvalTest:
  def __init__(self, model, cirterion, optimizer, scheduler, dataloaders, dataset_sizes, device, logger, model_name, learning_rate, batch_size):
    self.model = model
    self.criterion = cirterion
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.dataloaders = dataloaders
    self.dataset_sizes = dataset_sizes
    self.device = device
    self.logger = logger
    self.model_name = model_name
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.log_csv_filepath = '/home/matjaz.cigler1/Project1/Logs/trainValTest_log.csv'
    self.confusion_matrix_csv = '/home/matjaz.cigler1/Project1/Logs/confusion_matrix_log.csv'

  
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
            for inputs, labels in self.dataloaders[phase]:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = self.model(inputs)
                    i, predictions = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)
                    if phase == 'Train': #if training then change gradient
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    
                culm_loss += loss.item() * inputs.size(0)
                culm_correct += torch.sum(predictions == labels.data)
            
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
                    best_model = copy.deepcopy(self.model.state_dict())
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
                # self.logger.info('Loss per epoch (Training): {}'.format(train_loss_list))
                # self.logger.info('Accuracy per epoch(Training): {}'.format(train_acc_list))
                # self.logger.info('Loss per epoch (Validation): {}'.format(val_loss_list))
                # self.logger.info('Accuracy per epoch(Validation): {}'.format(val_acc_list))
                # load best model weights

                save_path = f"/home/matjaz.cigler1/Project1/Models/saved_models/{self.model_name}_lr_{self.learning_rate}_lr_{self.batch_size}"
                torch.save(best_model, save_path)
                self.model.load_state_dict(best_model)
                return self.model
                
            

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
    
    for inputs, labels in self.dataloaders['Test']:
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        with torch.set_grad_enabled(False):
            outputs = self.model(inputs)
            i, predictions = torch.max(outputs, 1)
            loss = self.criterion(outputs, labels)
        culm_loss += loss.item() * inputs.size(0)
        culm_correct += torch.sum(predictions == labels.data)

        all_predictions.extend(predictions.cpu().numpy())
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