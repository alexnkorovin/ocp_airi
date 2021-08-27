"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import pickle
import torch
from torch.utils.tensorboard import SummaryWriter

import wandb
from ocpmodels.common.registry import registry

from ocpmodels.datasets import SinglePointLmdbDataset, TrajectoryLmdbDataset

class Logger:
    """Generic class to interface with various logging modules, e.g. wandb,
    tensorboard, etc.
    """

    def __init__(self, config):
        self.config = config

    def watch(self, model):
        """
        Monitor parameters and gradients.
        """
        raise NotImplementedError

    def log(self, update_dict, step=None, split=""):
        """
        Log some values.
        """
        assert step is not None
        if split != "":
            new_dict = {}
            for key in update_dict:
                new_dict["{}/{}".format(split, key)] = update_dict[key]
            update_dict = new_dict
        return update_dict

    def log_plots(self, plots):
        raise NotImplementedError


@registry.register_logger("wandb")
class WandBLogger(Logger):
    def __init__(self, config):
        super().__init__(config)
        wandb.init(
            config=self.config,
            id=self.config["cmd"]["timestamp"],
            name=self.config["cmd"]["identifier"],
            dir=self.config["cmd"]["logs_dir"],
            project=self.config.get("logger_project", None),
        )

    def watch(self, model):
        wandb.watch(model)

    def log(self, update_dict, step=None, split=""):
        update_dict = super().log(update_dict, step, split)
        wandb.log(update_dict, step=int(step))

    def log_plots(self, plots, caption=""):
        assert isinstance(plots, list)
        plots = [wandb.Image(x, caption=caption) for x in plots]
        wandb.log({"data": plots})


@registry.register_logger("tensorboard")
class TensorboardLogger(Logger):
    def __init__(self, config):
        super().__init__(config)
        self.writer = SummaryWriter(self.config["cmd"]["logs_dir"])
        self.timestamp = self.config["cmd"]["logs_dir"].split("/")[-1]

    # TODO: add a model hook for watching gradients.
    def watch(self, model):
        pass
        #self.writer.add_graph(model, trace_element)
        #for m in module.children():
            #m.register_forward_hook(self.print_shape)
        return False

    def log(self, update_dict, step=None, split=""):
        update_dict = super().log(update_dict, step, split)
        for key in update_dict:
            if torch.is_tensor(update_dict[key]):
                self.writer.add_scalar(key, update_dict[key].item(), step)
            else:
                assert isinstance(update_dict[key], int) or isinstance(
                    update_dict[key], float
                )
                self.writer.add_scalar(key, update_dict[key], step)

    def send_hist(self, model, step=None, validation=True):
        for name, weight in model.named_parameters():
            #heatmaps
            if validation and step%15==0 and False:
                with open(f"logs/heatmaps/{self.timestamp}_{step}.pickle", "wb") as f:
                    pickle.dump(list(model.named_parameters()), f)
            #histograms
            self.writer.add_histogram(name, weight, step)