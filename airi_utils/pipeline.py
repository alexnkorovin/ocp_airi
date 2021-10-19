import importlib.util
import os

import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml

import numpy as np
import pandas as pd

from datetime import datetime
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, DataParallel
from torch_scatter import scatter
from torch.utils.tensorboard import SummaryWriter

from DataClasses import lmdb_dataset, Dataset, choose_dataloader
from ModelFunctions import multigpu_available, train, evaluate, inference
from ModelsNames import models_names

import sys
sys.path.append(os.path.expanduser('../ocpmodels/models'))
sys.path.append(os.path.expanduser('../../ocp_airi'))

def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument ('-c', '--config', nargs='?')
 
    return parser
 
 
if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    if namespace.config != None:
        with open(namespace.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError('No config filepath')

 
    # module_name = 'custom_spinconv'
    # file_path = os.path.expanduser('../ocpmodels/models/' + models_names[config['model_name']])

    # mod = importlib.load_source(module_name, file_path)

    module_name = 'custom_spinconv'
    file_path = os.path.expanduser('../ocpmodels/models/' + models_names[config['model_name']])
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    spinconv = mod.spinconv
    preprocessing = mod.preprocessing

    #config
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    lr = config['lr']
    epochs = config['epochs']

    features_cols = ['feature_1']
    target_col = 'y_relaxed'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(device)

#    train_dataset_file_path = os.path.expanduser("../../ocp_datasets/data/is2re/10k/train/data_mod1.lmdb")
    train_dataset_file_path = os.path.expanduser(config['train_dataset'])

    training_set = Dataset(train_dataset_file_path, features_cols, target_col, preprocessing=preprocessing)
    training_generator = choose_dataloader(training_set, batch_size=batch_size)

#    val_dataset_file_path = os.path.expanduser("../../ocp_datasets/data/is2re/all/val_ood_both/data_mod2.lmdbz")
    val_dataset_file_path = os.path.expanduser(config['val_dataset'])

    valid_set = Dataset(val_dataset_file_path, features_cols, target_col, preprocessing=preprocessing)
    valid_generator = choose_dataloader(valid_set, batch_size=batch_size, num_workers=num_workers)

    try:
        lmdb_dataset(train_dataset_file_path).describe()
    except:
        pass

    #model
    # model = spinconv(None, None, 1, otf_graph=False, regress_forces=False, use_pbc=False, cutoff_radii=10)
    model = spinconv(*config['model_args'], **config['model_kwargs'])
    if multigpu_available():
        model = DataParallel(model)

    #optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    #переносим на куду если она есть
    model = model.to(device)
    criterion = criterion.to(device)


    timestamp = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    print(timestamp)



    #tensorboard writer, при первом запуске надо руками сделать папку для логов

    # server
    #log_folder_path = "../../ocp_results/logs/tensorboard/out_base_model"

    # colab
    # log_folder_path = "/content/drive/MyDrive/ocp_results/logs/tensorboard/out_base_model"

    # user_specific 
    log_file_path = "../logs/tensorboard_airi"

    writer = SummaryWriter(log_file_path + '/' + timestamp)


    logfile_str = {
        "train_dataset_file_path": train_dataset_file_path,
        "val_dataset_file_path": val_dataset_file_path,
        "features_cols": features_cols,
        "target_col": target_col,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "epochs": epochs,
        "lr": lr,
        "type": config['model_name']
    }

    #граф модели
    try:
        #trace_system = dict(list(next(iter(training_generator))[0]))
        writer.add_graph(model, trace_system)
    except:
        print('no graph')
    writer.add_text(timestamp, str(logfile_str))

    loss = []
    loss_eval = []

    print(timestamp)
    print(f'Start training model {str(model)}')
    for i in range(epochs):
        loss.append(train(model, training_generator, optimizer, criterion, epoch=i, writer=writer, device=device))
        loss_eval.append(evaluate(model, valid_generator, criterion, epoch=i, writer=writer, device=device))




