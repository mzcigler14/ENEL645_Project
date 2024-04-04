import torch
import logging
import csv
from Models.ResNextModel import ResNextModel
from Models.SwinV2Model import SwinV2Model
from Models.Efficientnet_b0 import efficientnet_b0_model
from Models.Efficientnet_b1 import efficientnet_b1_model 
from Models.Efficientnet_b2 import efficientnet_b2_model 
from Models.Efficientnet_b3 import efficientnet_b3_model 
from Models.Efficientnet_b4 import efficientnet_b4_model 
from Models.Efficientnet_b5 import efficientnet_b5_model
from Models.Efficientnet_b6 import efficientnet_b6_model 
from Models.Efficientnet_b7 import efficientnet_b7_model 

from Models.CNN_New import CNNModel
from DataHandling import DataHandler

training_run_id = 1
project_directory ='/home/matjaz.cigler1/Project1/datasets'
grain_type = 'WR2021'

log_file_path = '/home/matjaz.cigler1/Project1/Logs/trainValTest_log.csv'
confusion_matrix_path = '/home/matjaz.cigler1/Project1/Logs/confusion_matrix_log.csv'

with open(log_file_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # Write the header row
    writer.writerow(['Model Name','Batch Size', 'Learning Rate', 'Phase', 'Epoch', 'Loss', 'Accuracy', 'Patience Count'])

with open(confusion_matrix_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # Write the header row
    writer.writerow(['Model Name','Batch Size', 'Learning Rate', 'Accuracy', 'Loss', 'Confusion Matrix'])

logger = logging.getLogger(f'Run{training_run_id}')
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(f'/home/matjaz.cigler1/Project1/Logs/runLog.log')
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) 

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)





logger.info(f'Starting Transfer Learn for Run {training_run_id}')


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

logger.info(f'Using device {device}\n')

#We need a datahandler to be shared between all runs so that the data in train/eval/test is the same

data_handler = DataHandler(project_directory, grain_type)

# batch_sizes = [8, 16, 32, 64, 128]
# learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]
batch_sizes = [8]
learning_rates = [0.01]

models_to_test = [
    SwinV2Model(device, logger, data_handler),
    ResNextModel(device, logger, data_handler),
    efficientnet_b0_model(device, logger, data_handler),
    efficientnet_b1_model(device, logger, data_handler),
    efficientnet_b2_model(device, logger, data_handler),
    efficientnet_b3_model(device, logger, data_handler),
    efficientnet_b4_model(device, logger, data_handler),
    efficientnet_b5_model(device, logger, data_handler),
    efficientnet_b6_model(device, logger, data_handler),
    efficientnet_b7_model(device, logger, data_handler),

    # CNNModel(device, logger, data_handler)
]

for model in models_to_test:
    for lr in learning_rates:
        for bs in batch_sizes:
            try:
                logger.info(f'Starting model train on {model.__class__.__name__} at learning Rate {lr} and batch size {bs}')
                model.prepare_model(lr, bs)
                best_model = model.train_model(30,5)
                model.test_model()
            except Exception as e:
                logger.error(f'An error occured within model {model.__class__.__name__} at Learning Rate {lr} and Batch size {bs}: Error {e}')
                continue



