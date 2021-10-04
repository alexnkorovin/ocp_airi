# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
import os
sys.path.append(os.path.expanduser('../ocpmodels/models'))
sys.path.append(os.path.expanduser('../ocpmodels'))

sys.path.append(os.path.expanduser('../ocp_airi'))
sys.path.append(os.path.expanduser('../airi_utils'))
sys.path.append(os.path.expanduser('.'))
sys.path.append(os.path.expanduser('..'))
sys.path.append(os.path.expanduser('../ocpmodels/models/'))

from DataClasses import lmdb_dataset, Dataset
from ModelFunctions import train, evaluate, inference

from IPython import get_ipython

# %%
import os

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torch.optim as optim



from datetime import datetime
from torch import nn
from torch_geometric.data import Data, DataLoader, DataListLoader
from torch_geometric.nn import MessagePassing, DataParallel
from torch_scatter import scatter
from torch.utils.tensorboard import SummaryWriter



# %%


from spinconv_with_val_vor_tag import spinconv


# %%
# sys.path.append(os.path.expanduser('/share'))
# from AIRIEmisisonTracker.AIRIEmisisonTracker import Tracker
# tracker = Tracker(project_name=your_project_name,
#                       experiment_description=your_experiment_description,
#                       save_file_name="you_file_name",
#                       measure_period=2,   #measurement will be done every 2 seconds
#                       emission_level=your_value,   #kg/MWTh
#                       base_power=your_gpu_base_power   #power of not working gpu
#                       )
# tracker.start()


# %%
#вызывается каждый раз, когда датасет отдаёт элемент (систему)
#делаем из данных матрицу векторов-атомов, список рёбер (edge_index) и матрицу векторов-рёбер; надо писать свою функцию для каждой сети
def preprocessing(system):
    keys = ['pos', 'atomic_numbers', 'cell', 'natoms', 'distances_new', 'voronoi_volumes', 'tags', 'edge_index_new']
    features_dict = {}
    for key in keys:
        features_dict[key] = system[key]
    return Data(**features_dict)


# %%
#config
batch_size = 30
num_workers = 0

features_cols = ['feature_1']

target_col = 'y_relaxed'
lr = 0.001
epochs = 50


# %%
# #чтобы тензор по умолчанию заводился на куде
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print('cuda')


# %%
#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
print(device)


# %%
#инициализируем тренировочный датасети и тренировочный итератор
train_dataset_file_path= os.path.expanduser("/share/catalyst/ocp_datasets/data/is2re/10k/train/data_mod.lmdb")

training_set = Dataset(train_dataset_file_path, features_cols, target_col, preprocessing=preprocessing)
training_generator = DataListLoader(training_set, batch_size=batch_size)


# %%
#инициализируем валидационный датасет и валидационный итератор
val_dataset_file_path = os.path.expanduser("/share/catalyst/ocp_datasets/data/is2re/all/val_ood_both/data_mod.lmdb")

valid_set = Dataset(val_dataset_file_path, features_cols, target_col, preprocessing=preprocessing)
valid_generator = DataListLoader(valid_set, batch_size=batch_size, num_workers=num_workers)


# %%
try:
    lmdb_dataset(train_dataset_file_path).describe()
except:
    pass


# %%
#model
model = spinconv(None, None, 1, otf_graph=True,use_pbc=True, regress_forces=False)
model = DataParallel(model)

#optimizer and loss
optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion = nn.L1Loss()

#переносим на куду если она есть
model = model.to(device)
criterion = criterion.to(device)


# %%
timestamp = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

print(timestamp)


# %%
#tensorboard writer, при первом запуске надо руками сделать папку для логов

# server
#log_folder_path = "../../ocp_results/logs/tensorboard/out_base_model"

# colab
# log_folder_path = "/content/drive/MyDrive/ocp_results/logs/tensorboard/out_base_model"

# user_specific 
log_file_path = "../../logs/tensorboard_airi"

writer = SummaryWriter(log_file_path + '/' + timestamp)


# %%
# get_ipython().run_cell_magic('time', '', 'logfile_str = {\n    "train_dataset_file_path": train_dataset_file_path,\n    "val_dataset_file_path": val_dataset_file_path,\n    "features_cols": features_cols,\n    "target_col": target_col,\n    "batch_size": batch_size,\n    "num_workers": num_workers,\n    "epochs": epochs,\n    "lr": lr\n}\n\n#граф модели\ntry:\n    #trace_system = dict(list(next(iter(training_generator))[0]))\n    writer.add_graph(model, trace_system)\nexcept:\n    print(\'no graph\')\nwriter.add_text(timestamp, str(logfile_str))')

# %% [markdown]
# ## Training

# %%
# get_ipython().run_cell_magic('time', '', "loss = []\nloss_eval = []\n\nfrom spinconv_with_radii import spinconv\n\nprint(timestamp)\nprint(f'Start training model {str(model)}')\nfor i in range(epochs):\n    loss.append(train(model, training_generator, optimizer, criterion, epoch=i, writer=writer, device=device))\n    loss_eval.append(evaluate(model, valid_generator, criterion, epoch=i, writer=writer, device=device))")
loss = []
loss_eval = []

from spinconv_with_radii import spinconv

print(timestamp)
print(f'Start training model {str(model)}')
for i in range(epochs):
    loss.append(train(model, training_generator, optimizer, criterion, epoch=i, writer=writer, device=device))
    loss_eval.append(evaluate(model, valid_generator, criterion, epoch=i, writer=writer, device=device))

# %%
# tracker.stop()


# %%



