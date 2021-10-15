"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import _pickle as pickle
import gc
import os
import pickle as pkl
import zlib
from typing import List, Union

import lmdb
import torch
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Dataset as pyg_Dataset

num_gpu = torch.cuda.device_count()

if num_gpu > 1:
    ### код прямиком из PyG чтобы не обновлять библиотеку


    def pickle_load_z(datapoint_pickled):
        return pickle.loads(zlib.decompress(datapoint_pickled))


    def pickle_load(datapoint_pickled):
        return pickle.loads(datapoint_pickled)


    class lmdb_dataset(Dataset):
        r"""Dataset class to load from LMDB files containing
        single point computations.
        Useful for Initial Structure to Relaxed Energy (IS2RE) task.

        Args:
            config (dict): Dataset configuration
            transform (callable, optional): Data transform function.
                (default: :obj:`None`)
        """

        def __init__(
            self, config, transform=None, compressed=False, multiproc=False, byte=False
        ):
            super().__init__()

            self.config = config
            self.multiproc = multiproc
            self.byte = byte
            self.db_path = self.config

            assert os.path.isfile(self.db_path), "{} not found".format(
                self.db_path
            )

            self.compressed = compressed

            suffix = self.db_path.split(".")[-1]

            self.compressed = True if suffix == "lmdbz" else False

            self.env = self.connect_db(self.db_path)

            # # key by number of elements - faster bu only for keys in range(0, num)
            # self.keys = [
            #     f"{j}".encode("ascii") for j in range(self.env.stat()["entries"])
            # ]

            # keys by key names
            with self.env.begin() as txn:
                with txn.cursor() as curs:
                    self.keys = [key for key, value in curs]

            #        self.keys = [f"{j}".encode("ascii") for j in txn.cursor().iternext(key=True, value=False)]
            self.transform = transform

        def __len__(self):
            return len(self.keys)

        def new(self):
            return lmdb_dataset(self.config, self.transform, self.compressed, self.multiproc, self.byte)

        def iloc(self, start, stop):
            self.keys = self.keys[start:stop]

        def __getstate__(self):
            # this method is called when you are
            # going to pickle the class, to know what to pickle
            state = self.__dict__.copy()

            # don't pickle the parameter env
            del state["env"]

            return state

        def __setstate__(self, state):
            self.__dict__.update(state)
            # retrieve the excluded env
            self.env = self.connect_db(self.db_path)

        def __getitem__(self, idx):
            # Return features.
            if type(idx) is int:
                datapoint_pickled = self.env.begin().get(self.keys[idx])
            elif type(idx) is str:
                datapoint_pickled = self.env.begin().get(idx.encode('ascii'))

            gc.disable()

            if self.byte == True:
                data_object = datapoint_pickled
            else:
                if self.multiproc is False:
                    data_object = (
                        pickle.loads(zlib.decompress(datapoint_pickled))
                        if self.compressed is True
                        else pickle.loads(datapoint_pickled)
                    )
                else:
                    data_object = (
                        pickle.loads(zlib.decompress(datapoint_pickled))
                        if self.compressed is True
                        else pickle.loads(datapoint_pickled)
                    )
                    # data_object = Parallel(n_jobs=2)(delayed(restore_edge_angles)(el['edge_angles']) for el in dataset_target)

            data_object = (
                data_object
                if self.transform is None
                else self.transform(data_object)
            )
            gc.enable()
            # print(self.env.info())
            return data_object

        def connect_db(self, lmdb_path=None):

            env = lmdb.open(
                lmdb_path,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=1000,
            )
            # print(env.info())
            return env

        def close_db(self):
            self.env.close()

        def describe(self, idx=0):
            self.idx = idx
            print(f'total entries: {self.env.stat()["entries"]}')

            dataset = self.__getitem__(idx)
            print(f"info for item: {self.idx}")

            try:
                keys = dataset.keys()

            except TypeError:
                keys = dataset.keys

            for key in keys:
                # print(type(dataset[key]))
                obj = dataset[key]
                dot = 25

                if "torch.Tensor" in str(type(obj)):
                    print(
                        f'{key}:{"." * (dot - len(key))}{str(type(obj)):>20}{"." * 5}{str(list(obj.shape)):>10}'
                    )
                elif type(obj) is float:
                    print(
                        f'{key}:{"." * (dot - len(key))}{str(type(obj)):>20}{"." * 5}{obj:>10.4f}'
                    )

                elif type(obj) is int:
                    print(
                        f'{key}:{"." * (dot - len(key))}{str(type(obj)):>20}{"." * 5}{obj:>10}'
                    )

                elif type(obj) is list:
                    print(
                        f'{key}:{"." * (dot - len(key))}{str(type(obj)):>20}{"." * 5}{len(obj):>10}'
                    )

                elif type(obj) is list:
                    print(
                        f'{key}:{"." * (dot - len(key))}{str(type(obj)):>20}{"." * 5}{len(obj):>10}'
                    )

                elif type(obj) is str:
                    print(
                        f'{key}:{"." * (dot - len(key))}{str(type(obj)):>20}{"." * 5}{len(obj):>10}'
                    )

                else:
                    print(
                        f'{key}:{"." * (dot - len(key))}{str(type(obj)):>20}{"." * 5}{type(obj):>10}'
                    )

        def info(self):
            print(self.env.info())

        def set_map_size(self, map_size):
            self.env.set_mapsize(map_size)
            print(f'map_size: {self.env.info()["map_size"]}')

        def stat(self):
            print(self.env.stat())


    class Dataset(pyg_Dataset):
        def __init__(
            self,
            data,
            features_fields,
            target_field,
            type_="train",
            preprocessing=None,
        ):

            self.data = data
            self.length = len(self.data)
            # self.target = data[target_field]
            self.type_ = type_
            self.preprocessing = preprocessing
            self.target = target_field

        def __len__(self):
            return self.length

        def __getitem__(self, index):

            system = self.preprocessing(self.data[index])

            if self.type_ == "train":
                system.y = torch.tensor(self.data[index][self.target]).reshape(
                    (1, 1)
                )

            return system


    def collate_fn(data_list):
        return data_list


    class DataListLoader(torch.utils.data.DataLoader):
        r"""A data loader which batches data objects from a
        :class:`torch_geometric.data.dataset` to a Python list.
        Data objects can be either of type :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData`.
        .. note::
            This data loader should be used for multi-GPU support via
            :class:`torch_geometric.nn.DataParallel`.
        Args:
            dataset (Dataset): The dataset from which to load the data.
            batch_size (int, optional): How many samples per batch to load.
                (default: :obj:`1`)
            shuffle (bool, optional): If set to :obj:`True`, the data will be
                reshuffled at every epoch. (default: :obj:`False`)
            **kwargs (optional): Additional arguments of
                :class:`torch.utils.data.DataLoader`, such as :obj:`drop_last` or
                :obj:`num_workers`.
        """

        def __init__(
            self,
            dataset: Union[pyg_Dataset, List[Data]],
            batch_size: int = 1,
            shuffle: bool = False,
            **kwargs,
        ):
            if "collate_fn" in kwargs:
                del kwargs["collate_fn"]

            super().__init__(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=collate_fn,
                **kwargs,
            )
else:
    def pickle_load_z(datapoint_pickled):
        return pickle.loads(zlib.decompress(datapoint_pickled))


    def pickle_load(datapoint_pickled):
        return pickle.loads(datapoint_pickled)


    class lmdb_dataset(Dataset):
        r"""Dataset class to load from LMDB files containing
        single point computations.
        Useful for Initial Structure to Relaxed Energy (IS2RE) task.

        Args:
            config (dict): Dataset configuration
            transform (callable, optional): Data transform function.
                (default: :obj:`None`)
        """

        def __init__(
                self, config, transform=None, compressed=False, multiproc=False
        ):
            super().__init__()

            self.config = config
            self.multiproc = multiproc

            self.db_path = self.config

            assert os.path.isfile(self.db_path), "{} not found".format(
                self.db_path
            )

            self.compressed = compressed

            suffix = self.db_path.split(".")[-1]

            self.compressed = True if suffix == "lmdbz" else False

            self.env = self.connect_db(self.db_path)

            self._keys = [
                f"{j}".encode("ascii") for j in range(self.env.stat()["entries"])
            ]

            #        self._keys = [f"{j}".encode("ascii") for j in txn.cursor().iternext(key=True, value=False)]
            self.transform = transform

        def __len__(self):
            return len(self._keys)

        def __getstate__(self):
            # this method is called when you are
            # going to pickle the class, to know what to pickle
            state = self.__dict__.copy()

            # don't pickle the parameter env
            del state["env"]

            return state

        def __setstate__(self, state):
            self.__dict__.update(state)
            # retrieve the excluded env
            self.env = self.connect_db(self.db_path)

        def __getitem__(self, idx):
            # Return features.
            datapoint_pickled = self.env.begin().get(self._keys[idx])
            gc.disable()
            if self.multiproc is False:
                data_object = (
                    pickle.loads(zlib.decompress(datapoint_pickled))
                    if self.compressed is True
                    else pickle.loads(datapoint_pickled)
                )
            else:
                data_object = (
                    pickle.loads(zlib.decompress(datapoint_pickled))
                    if self.compressed is True
                    else pickle.loads(datapoint_pickled)
                )
            # a = Parallel(n_jobs=2)(delayed(restore_edge_angles)(el['edge_angles']) for el in dataset_target)

            data_object = (
                data_object
                if self.transform is None
                else self.transform(data_object)
            )
            gc.enable()
            # print(self.env.info())
            return data_object

        def connect_db(self, lmdb_path=None):

            env = lmdb.open(
                lmdb_path,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=1000,
            )
            # print(env.info())
            return env

        def close_db(self):
            self.env.close()

        def describe(self, idx=0):
            self.idx = idx
            print(f'total entries: {self.env.stat()["entries"]}')

            dataset = self.__getitem__(idx)
            print(f"info for item: {self.idx}")

            try:
                keys = dataset.keys()

            except TypeError:
                keys = dataset.keys

            for key in keys:
                # print(type(dataset[key]))
                obj = dataset[key]
                dot = 25

                if "torch.Tensor" in str(type(obj)):
                    print(
                        f'{key}:{"." * (dot - len(key))}{str(list(obj.shape)):>10}'
                    )
                elif type(obj) is float:
                    print(f'{key}:{"." * (dot - len(key))}{obj:>10.4f}')

                elif type(obj) is int:
                    print(f'{key}:{"." * (dot - len(key))}{obj:>10}')

                elif type(obj) is list:
                    print(f'{key}:{"." * (dot - len(key))}{len(obj):>10}')

                else:
                    print(f'{key}:{"." * (dot - len(key))}{type(obj) > 10}')

        def info(self):
            print(self.env.info())

        def set_map_size(self, map_size):
            self.env.set_mapsize(map_size)
            print(f'map_size: {self.env.info()["map_size"]}')

        def stat(self):
            print(self.env.stat())


    class Dataset(pyg_Dataset):
        def __init__(
                self,
                data,
                features_fields,
                target_field,
                type_="train",
                preprocessing=None,
        ):
            self.data = lmdb_dataset(data)
            self.length = len(self.data)
            # self.target = data[target_field]
            self.type_ = type_
            self.preprocessing = preprocessing
            self.target = target_field

        def __len__(self):
            return self.length

        def __getitem__(self, index):
            system = self.preprocessing(self.data[index])

            if self.type_ == "train":
                y = self.data[index][self.target]

                return system, y

    class DataListLoader:
        def __init__(*args, **kwargs):
            raise ValueError('Для обучения на cpu или одной gpu нужно использовать DataLoader')