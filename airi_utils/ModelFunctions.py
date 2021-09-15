import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from DataClasses import Dataset, lmdb_dataset
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter


def send_scalars(lr, loss, writer, step=-1, epoch=-1, type_="train"):
    if type_ == "train":
        writer.add_scalar("lr per step on train", lr, step)
        writer.add_scalar("loss per step on train", loss, step)
    if type_ == "val":
        writer.add_scalar("loss per epoch on val", loss, epoch)


def send_hist(model, writer, step):
    for name, weight in model.named_parameters():
        writer.add_histogram(name, weight, step)


def train(
    model,
    iterator,
    optimizer,
    criterion,
    print_every=10,
    epoch=0,
    writer=None,
    device="cpu",
):

    print(f"epoch {epoch}")

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


def evaluate(model, iterator, criterion, epoch=0, writer=False, device="cpu"):

    print(f"epoch {epoch} evaluation")

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
    print(
        "========================================================================================================"
    )

    return overall_loss


def inference(model, iterator):
    y = torch.tensor([])

    #    model.train(False)
    model.eval()

    with torch.no_grad():
        for systems, ys in iterator:
            predictions = model(systems).squeeze()
            y = torch.cat((y, predictions))

    return y


def my_reshape(tensor):
    return torch.reshape(tensor, (tensor.shape[0], 1))


# convert list of thetas in edge_angles to bins
def to_bins_torch(array_of_dfs):
    thetas = []

    for df in array_of_dfs:
        theta = torch.tensor(df[:, 1])  # .to('cpu')
        theta = torch.histc(theta, bins=10, min=0, max=np.pi)
        theta = torch.reshape(theta, (1, theta.shape[0]))
        thetas.append(theta)

    thetas = torch.cat(thetas, 0)

    return thetas.float()


# restore full list of edge_angles data for ji based on ij
def convert_angles(array):
    array[:, 1] = np.pi - array[:, 1]
    array[:, 3] = -array[:, 3]
    return array


def restore_edge_angles(list_of_arrays):
    el_new = []
    for el in list_of_arrays:
        el_new.append(el)
        el_new.append(convert_angles(el.copy()))
    return el_new


# вызывается каждый раз, когда датасет отдаёт элемент (систему)
# делаем из данных матрицу векторов-атомов, список рёбер (edge_index) и матрицу векторов-рёбер; надо писать свою функцию для каждой сети
def preprocessing(system, opt="angles"):

    device = device_set()
    tags = system["tags"].long()
    tags = F.one_hot(tags, num_classes=3)

    atom_numbers = system["atomic_numbers"].long()
    atom_numbers = F.one_hot(atom_numbers, num_classes=100)

    voronoi_volumes = system["voronoi_volumes"].float()
    voronoi_volumes = my_reshape(voronoi_volumes)

    atom_features = (tags, atom_numbers, voronoi_volumes)  # , spherical_radii)
    atom_embeds = torch.cat(atom_features, 1)

    edge_index = system["edge_index_new"].long()

    distances = system["distances_new"].float()
    distances = my_reshape(distances)

    if opt == "angles":
        thetas = to_bins_torch(restore_edge_angles(system["edge_angles"]))
        #     angles = system['contact_solid_angles'].float().to(device)
        #     angles = my_reshape(angles)
        edge_features = (distances, thetas)

    else:
        edge_features = (distances,)

    edges_embeds = torch.cat(edge_features, 1)
    return Data(
        x=atom_embeds.to(device),
        edge_index=edge_index.to(device),
        edge_attr=edges_embeds.to(device),
    )


# set device
def device_set():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
