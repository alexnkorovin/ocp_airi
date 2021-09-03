import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, DataLoader, Dataset
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter

# CONFIG ------------------------------------------------------
voronoi_volumes_str = "voloroi_volumes"

# for colab
# train_dataset_file_path = "/content/drive/MyDrive/ocp_datasets/data/is2re/10k/train/structures_train.pkl"


# user specific folder
train_dataset_file_path = os.path.expanduser(
    "../../ocp_datasets/data/is2re/10k/train/structures.pkl"
)
# train_dataset_file_path= "../../ocp_datasets/data/is2re/10k/train/structures_tain.pkl"

val_dataset_file_path = os.path.expanduser(
    "../../ocp_datasets/data/is2re/all/val_ood_both/structures.pkl"
)
# val_dataset_file_path = os.path.expanduser("~/Downloads/structures_train.pkl")


# features_cols = ['voloroi_volumes', 'voronoi_surface_areas', 'electronegativity',
#                  'dipole_polarizability', 'edge_index_new', 'distances_new', 'contact_solid_angles']

# features_cols = ['atomic_numbers', 'edge_index_new', 'distances_new',
#                  'contact_solid_angles', 'tags', 'voronoi_volumes', 'spherical_domain_radii']
# target_col = 'y_relaxed'

features_cols = [
    "atomic_numbers",
    "edge_index_new",
    "distances_new",
    "contact_solid_angles",
    "tags",
    voronoi_volumes_str,
    "spherical_domain_radii",
]
target_col = "y_relaxed"

batch_size = 256
num_workers = 0
epochs = 50

logfile_str = {
    "train_dataset_file_path": train_dataset_file_path,
    "val_dataset_file_path": val_dataset_file_path,
    "features_cols": features_cols,
    "target_col": target_col,
    "batch_size": batch_size,
    "num_workers": num_workers,
    "epochs": epochs,
}
# -----------------------------------------------------


def my_reshape(tensor):
    return torch.reshape(tensor, (tensor.shape[0], 1))


# делаем из данных матрицу векторов-атомов, список рёбер (edge_index) и матрицу векторов-рёбер
def simple_preprocessing(batch):
    # spherical_radii = torch.Tensor(batch['spherical_domain_radii'])
    # spherical_radii = my_reshape(spherical_radii)

    tags = batch["tags"].long().to(device)
    tags = F.one_hot(tags, num_classes=3)

    atom_numbers = batch["atomic_numbers"].long().to(device)
    atom_numbers = F.one_hot(atom_numbers, num_classes=100)

    voronoi_volumes = torch.Tensor(batch[voronoi_volumes_str])
    voronoi_volumes = my_reshape(voronoi_volumes)

    atom_features = (tags, atom_numbers, voronoi_volumes)  # , spherical_radii)
    atom_embeds = torch.cat(atom_features, 1)

    edge_index = torch.Tensor(batch["edge_index_new"]).long()

    distances = torch.Tensor(batch["distances_new"])
    distances = my_reshape(distances)

    angles = torch.Tensor(batch["contact_solid_angles"])
    angles = my_reshape(angles)

    edge_features = (distances, angles)

    edges_embeds = torch.cat(edge_features, 1)

    return Data(
        x=atom_embeds.to(device),
        edge_index=edge_index.to(device),
        edge_attr=edges_embeds.to(device),
    )


# датасет, который умеет возвращать эелемент и собственную длину
class Dataset(Dataset):
    def __init__(
        self,
        data,
        features_fields,
        target_field,
        type_="train",
        preprocessing=simple_preprocessing,
    ):
        self.data = data[features_fields]
        self.length = len(data)
        self.target = torch.Tensor(data[target_field].values)
        self.type_ = type_
        self.preprocessing = preprocessing

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        system = self.preprocessing(self.data.iloc[index])

        if self.type_ == "train":
            y = self.target[index]

            return system, y


class GConv(MessagePassing):
    def __init__(self, dim_atom=103, dim_edge=2, out_channels=2):
        super(GConv, self).__init__(aggr="add")  # "Add" aggregation
        self.phi_output = 3
        self.lin_phi = torch.nn.Linear(
            dim_atom * 2 + dim_edge, self.phi_output, bias=False
        )
        self.lin_gamma = torch.nn.Linear(
            dim_atom + self.phi_output, out_channels, bias=False
        )
        self.nonlin = nn.Sigmoid()

    def forward(self, batch):
        x = batch["x"]
        edge_index = batch["edge_index"]
        edge_attr = batch["edge_attr"]

        # x has shape [N -- количество атомов в системе(батче), in_channels -- размерность вектора-атома]
        # edge_index has shape [2, E] -- каждое ребро задаётся парой вершин

        # Start propagating messages.

        return self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=None
        )  # не совсем понял что такое сайз

    def message(self, x, x_i, x_j, edge_attr):
        concatenated = torch.cat((x_i, x_j, edge_attr), 1)
        phi = self.lin_phi(concatenated)
        phi = self.nonlin(phi)
        return phi

    def update(self, aggr_out, x, edge_attr, edge_index):
        concatenated = torch.cat((x, aggr_out), 1)
        gamma = self.lin_gamma(concatenated)
        gamma = self.nonlin(gamma)

        return Data(x=gamma, edge_attr=edge_attr, edge_index=edge_index)


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class InteractionBlock(nn.Module):
    def __init__(self, dim_atom, dim_edge, num_filters):
        super(InteractionBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_edge, dim_atom),
            ShiftedSoftplus(),
            nn.Linear(dim_atom, num_filters),
        )
        self.conv = CFConv(dim_atom, dim_atom, num_filters, self.mlp)
        self.act = ShiftedSoftplus()
        self.lin = nn.Linear(dim_atom, num_filters)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv(x, edge_index, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, n_n):
        super(CFConv, self).__init__(aggr="add")
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, out_channels)
        self.nn = n_n

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        W = self.nn(edge_attr)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


# собственно нейросеть
class ConvNN(nn.Module):
    def __init__(self, dim_atom=103, dim_edge=2, out_channels=50):
        super().__init__()

        self.conv_1 = InteractionBlock(
            dim_atom=dim_atom, dim_edge=dim_edge, num_filters=dim_atom
        )

        self.conv_last = InteractionBlock(
            dim_atom=dim_atom, dim_edge=dim_edge, num_filters=out_channels
        )

        self.lin = torch.nn.Linear(out_channels, 1, bias=True)

    def forward(self, batch):
        convoluted_1 = self.conv_1(
            batch["x"], batch["edge_index"], batch["edge_attr"]
        )
        convoluted_last = self.conv_last(
            convoluted_1, batch["edge_index"], batch["edge_attr"]
        )
        scattered = scatter(
            convoluted_last, batch["batch"], dim=0, reduce="sum"
        )
        summed = scattered
        energy = self.lin(summed)

        return energy


def send_scalars(lr, loss, writer, step=-1, epoch=-1, type_="train"):
    if type_ == "train":
        writer.add_scalar("lr per step on train", lr, step)
        writer.add_scalar("loss per step on train", loss, step)
    if type_ == "val":
        writer.add_scalar("loss per epoch on val", loss, epoch)


def send_hist(model, writer, step):
    for name, weight in model.named_parameters():
        writer.add_histogram(name, weight, step)


# train -- ходим по батчам из итератора, обнуляем градиенты, предсказываем у, считаем лосс, считаем градиенты, делаем шаг оптимайзера, записываем лосс
def train(
    model, iterator, optimizer, criterion, print_every=10, epoch=0, writer=None
):
    epoch_loss = 0

    model.train()

    for i, (systems, ys) in enumerate(iterator):

        optimizer.zero_grad()
        predictions = model(systems).squeeze()

        loss = criterion(predictions.float(), ys.to(device).float())
        loss.backward()

        optimizer.step()

        batch_loss = loss.item()
        epoch_loss += batch_loss

        if writer is not None:
            lr = optimizer.param_groups[0]["lr"]

            step = i + epoch * len(iterator)

            send_hist(model, writer, i)
            send_scalars(
                lr, batch_loss, writer, step=step, epoch=epoch, type_="train"
            )

        if not (i + 1) % print_every:
            print(f"step {i} from {len(iterator)} at epoch {epoch}")
            print(f"Loss: {batch_loss}")

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, epoch=0, writer=False):
    epoch_loss = 0

    #    model.train(False)
    model.eval()

    with torch.no_grad():
        for systems, ys in iterator:
            predictions = model(systems).squeeze()
            loss = criterion(predictions.float(), ys.to(device).float())

            epoch_loss += loss.item()

    overall_loss = epoch_loss / len(iterator)

    if writer is not None:
        send_scalars(
            None, overall_loss, writer, step=None, epoch=epoch, type_="val"
        )

    print(f"epoch loss {overall_loss}")

    return overall_loss


def inferens(model, iterator):
    y = torch.tensor([])

    #    model.train(False)
    model.eval()

    with torch.no_grad():
        for systems, ys in iterator:
            predictions = model(systems).squeeze()
            y = torch.cat((y, predictions))

    return y


def read_df(filename):
    with open(filename, "rb") as f:
        data_ori = pickle.load(f)

    # сливаем новые фичи и фичи из Data
    for system in data_ori:
        for key in system["data"]:
            system[key[0]] = key[1]
        del system["data"]

    df = pd.DataFrame(data_ori)
    data_ori = []
    print(df.columns)
    return df


df_train = read_df(train_dataset_file_path)
df_val = read_df(val_dataset_file_path)

# инициализируем тренировочный датасети и тренировочный итератор
training_set = Dataset(df_train, features_cols, target_col)
training_generator = DataLoader(
    training_set, batch_size=batch_size, num_workers=num_workers
)

# инициализируем валидационный датасет и валидационный итератор
valid_set = Dataset(df_val, features_cols, target_col)
valid_generator = DataLoader(
    valid_set, batch_size=batch_size, num_workers=num_workers
)

df_train = []
df_val = []

# чтобы тензор по умолчанию заводился на куде
if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print("cuda")

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# model
model = ConvNN(
    dim_atom=training_set[0][0].x.shape[1],
    dim_edge=training_set[0][0].edge_attr.shape[1],
)

# optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.L1Loss()

# переносим на куду если она есть
model = model.to(device)
criterion = criterion.to(device)

timestamp = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

print(timestamp)

# tensorboard writer, при первом запуске надо руками сделать папку для логов

# server
# log_folder_path = "../../ocp_results/logs/tensorboard/out_base_model"

# colab
# log_folder_path = "/content/drive/MyDrive/ocp_results/logs/tensorboard/out_base_model"

# user_specific
log_file_path = "../logs/tensorboard_airi"

writer = SummaryWriter(log_file_path + "/" + timestamp)

# граф модели
trace_system = dict(list(next(iter(training_generator))[0]))
writer.add_graph(model, trace_system)
writer.add_text(timestamp, str(logfile_str))

time_now = datetime.now()
loss = []
loss_eval = []

print(timestamp)
print(f"Start training model {str(model)}")
for i in range(epochs):
    print(f"epoch {i}")
    loss.append(
        train(
            model,
            training_generator,
            optimizer,
            criterion,
            epoch=i,
            writer=writer,
        )
    )
    loss_eval.append(
        evaluate(model, valid_generator, criterion, epoch=i, writer=writer)
    )

writer.close()

print(
    f"Done for, s: {(datetime.now() - time_now).seconds} s; Loss: {min(loss_eval)}"
)
