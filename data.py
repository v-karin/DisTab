import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from torch import nn, Tensor
import numpy as np
from typing import Any, Literal
import os
import pandas as pd
from sklearn import preprocessing   
from sklearn.model_selection import train_test_split

from dataclasses import dataclass
import pickle
import random

import openml
from utils import TASK_ID

import gin

def init_worker_fn():
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def custom_collate(batch):
    header_info = batch[0][0]
    tmp = [item[1:] for item in batch]
    out = default_collate(tmp)
    x, y = out
    return header_info, x, y


def get_tab_data(data_name, data_root):
    data_path = f"{data_root}/{data_name}/tab_data"

    with open(data_path, "rb") as fd:
        tab_data = pickle.load(fd)
    return tab_data

def process_cat_cols(df, cat_embed):
    cat_df = df.loc[:, df.dtypes == "string"]
    if cat_df.empty:
        return None, None
    cat_np = cat_df.values
    ord_encoder = preprocessing.OrdinalEncoder()
    ord_encoder.fit(cat_np)

    categories = ord_encoder.categories_

    flat_embed = []
    for i, col_name in enumerate(cat_df.columns):
        encoder_cat = categories[i]
        col_embed_map = cat_embed[col_name]
        assert len(encoder_cat) == len(col_embed_map)

        for cell_val in encoder_cat:
            flat_embed.append(col_embed_map[cell_val])
    
    return np.concatenate(flat_embed), ord_encoder


def get_openml_task(data_name):
    task_id = TASK_ID[data_name]
    task = openml.tasks.get_task(task_id, download_data=False, download_qualities=False, download_splits=False, download_features_meta_data=False)
    return task


def arg_kth(arr, k):
    return np.argpartition(arr, k)[k]


def kth_largest(arr, k):
    if isinstance(k, float) and k <=1:
        k = int(len(arr) * k)
    ind = arg_kth(arr, k)
    return arr[ind]


def merge_data(real_x, real_y, new_x, new_y):
    tmp_x = pd.concat([real_x, new_x], axis=0)
    tmp_y = pd.concat([real_y, new_y], axis=0)
    tmp_x = tmp_x.reset_index(drop=True) 
    tmp_y = tmp_y.reset_index(drop=True) 
    return tmp_x, tmp_y


@gin.configurable
def get_synthetic_data_splits(tree_type_metric, model_dir, data_name, task, tab_data, fold=0, embed_offset=True, pct=0.12, seed=0, y_transform=True):
    from autogluon.tabular import TabularPredictor
    train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=fold)
    df, cat_dim_info, _, task_type, label_header = tab_data["df_data"]

    num_headers = []
    cat_headers = []

    for col in df.columns:
        if df[col].dtype == "float32":
            num_headers.append(col)
        else:
            cat_headers.append(col)

    train_X, test_X = df.iloc[train_indices], df.iloc[test_indices]
    train_X = train_X.reset_index(drop=True)
    test_X = test_X.reset_index(drop=True)
    flat_cat_embed, x_cat_encoder = process_cat_cols(df.drop(columns=[label_header]), tab_data["cat_embed"])

    metric = tree_type_metric[task_type]

    norm_tpl = train_X.copy(deep=True)
    valid_size = min(int(pct * train_X.shape[0]), 2500)
    if task_type == "regression":
        norm_tpl, _ = train_test_split(norm_tpl, test_size=valid_size, random_state=seed)
    else:
        norm_tpl, _ = train_test_split(train_X, test_size=valid_size, random_state=seed, stratify=train_X[label_header])

    norm_tpl = norm_tpl.reset_index(drop=True)
    
    tmp = [norm_tpl, train_X, test_X]

    for col in df.columns:
        if df[col].dtype == "float32":
            for e in tmp:
                e[col] = e[col].fillna(norm_tpl.loc[:, col].mean())
    
    soft_label = True

    model_folder = f"./{model_dir}/{data_name}_{fold}_{metric}/"
    if not os.path.exists(model_folder):
        raise FileNotFoundError(
            f"No such file or directory: '{model_folder}'. "
            "Please re-run the teacher model for the task with the required number of folds."
        )
    tree_predictor = TabularPredictor.load(model_folder)
    synthetic_db = SyntheticDF(train_X, label_header, tree_predictor, task_type, seed=seed)

    aug_size = 100000

    new_df, new_y = synthetic_db.batch_synthetic(aug_size, soft=soft_label)

    pred_y = synthetic_db._predict(train_X, soft=soft_label)

    orig_X, orig_y = split_y_off(train_X, label_header)
    num_labels = len(np.unique(orig_y.values))

    train_dataset = DynaLMTabData(norm_tpl, flat_cat_embed, x_cat_encoder, tab_data["header_embed"],
                                    cat_dim_info, data_name, task_type, label_header, num_labels, embed_offset=embed_offset,
                                    header_names=(num_headers, cat_headers), augment=0.)

    new_df, new_y = merge_data(orig_X, pred_y, new_df, new_y)
    aug_dataset = DynaLMTabData((new_df, new_y), flat_cat_embed, x_cat_encoder, tab_data["header_embed"],
                                    cat_dim_info, data_name, task_type, label_header,
                                    num_labels,
                                    y_encoder=train_dataset.y_encoder,
                                    embed_offset=embed_offset,
                                    header_names=(num_headers, cat_headers), augment=0.,
                                    train_db=True)

    test_dataset = DynaLMTabData(test_X, flat_cat_embed, x_cat_encoder, tab_data["header_embed"], cat_dim_info, data_name, 
                                        task_type, label_header, num_labels, y_encoder=train_dataset.y_encoder, embed_offset=embed_offset)

    train_dataset.normalize_num()
    if hasattr(train_dataset, "num_encoder"):
        aug_dataset.normalize_num(train_dataset.num_encoder)
        test_dataset.normalize_num(train_dataset.num_encoder)

    if train_dataset.task_type == "regression":
        y_normalizer = train_dataset.gen_y_normalizer("standard")
        aug_dataset.set_y_normalizer(y_normalizer, transform=y_transform)
        test_dataset.set_y_normalizer(y_normalizer, transform=y_transform)
    
    return aug_dataset, train_dataset, test_dataset


def get_data_splits(data_name, task, tab_data, fold=0, embed_offset=True, pct=0.12, seed=0, augment=0., y_transform=True):
    train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=fold)
    df, cat_dim_info, _, task_type, label_header = tab_data["df_data"]

    num_headers = []
    cat_headers = []

    for col in df.columns:
        if df[col].dtype == "float32":
            num_headers.append(col)
        else:
            cat_headers.append(col)

    train_X, test_X = df.iloc[train_indices], df.iloc[test_indices]
    train_X = train_X.reset_index(drop=True)
    test_X = test_X.reset_index(drop=True)
    flat_cat_embed, x_cat_encoder = process_cat_cols(df.drop(columns=[label_header]), tab_data["cat_embed"])
    valid_size = min(int(pct * train_X.shape[0]), 2500)

    def split_by_seed(train_X, seed, trial=True):
        train_X = train_X.copy(deep=True)
        if task_type == "regression":
            train_X, valid_X = train_test_split(train_X, test_size=valid_size, random_state=seed)
        else:
            train_X, valid_X = train_test_split(train_X, test_size=valid_size, random_state=seed, stratify=train_X[label_header])
    
        train_X = train_X.reset_index(drop=True)
        valid_X = valid_X.reset_index(drop=True)
    
        if trial:
            tmp = [train_X, valid_X]
        else:
            tmp = [train_X, valid_X, test_X]

        for col in df.columns:
            if df[col].dtype == "float32":
                for e in tmp:
                    e[col] = e[col].fillna(train_X.loc[:, col].mean())
    
        train_dataset = DynaLMTabData(train_X, flat_cat_embed, x_cat_encoder, tab_data["header_embed"],
                                      cat_dim_info, data_name, task_type, label_header, embed_offset=embed_offset,
                                      header_names=(num_headers, cat_headers), augment=augment, synthetic_df=None, train_db=True)

        valid_dataset = DynaLMTabData(valid_X, flat_cat_embed, x_cat_encoder, tab_data["header_embed"], cat_dim_info, data_name, 
                                    task_type, label_header, y_encoder=train_dataset.y_encoder, embed_offset=embed_offset)
        
        if trial and train_dataset.task_type == "regression":
            y_normalizer = train_dataset.gen_y_normalizer("standard")
            valid_dataset.set_y_normalizer(y_normalizer)
        if not trial:
            test_dataset = DynaLMTabData(test_X, flat_cat_embed, x_cat_encoder, tab_data["header_embed"], cat_dim_info, data_name, 
                                        task_type, label_header, y_encoder=train_dataset.y_encoder, embed_offset=embed_offset)

        if trial:
            return train_dataset, valid_dataset
        else:
            return train_dataset, valid_dataset, test_dataset
    
    best_seed = seed

    train_dataset, valid_dataset, test_dataset = split_by_seed(train_X, best_seed, trial=False)

    train_dataset.normalize_num()
    if hasattr(train_dataset, "num_encoder"):
        valid_dataset.normalize_num(train_dataset.num_encoder)
        test_dataset.normalize_num(train_dataset.num_encoder)

    if train_dataset.task_type == "regression":
        y_normalizer = train_dataset.gen_y_normalizer("standard")
        valid_dataset.set_y_normalizer(y_normalizer, transform=y_transform)
        test_dataset.set_y_normalizer(y_normalizer, transform=y_transform)

    return train_dataset, valid_dataset, test_dataset

def split_y_off(df, label_header):
    y = df[label_header]
    x = df.drop(columns=[label_header])
    return x, y


class SyntheticDF(object):
    def __init__(self, data_df, label_header, predictor, task_type, seed=0):
        super().__init__()
        data_df, y = split_y_off(data_df, label_header)
        self.label_header = label_header
        self.data_df = data_df
        self.y = y
        self.predictor = predictor
        self.task_type = task_type
        self.rng = np.random.RandomState(seed)
        self.count = len(data_df)
        self.col_count = data_df.shape[1]

        
    def _predict(self, sample, soft=True):
        if self.task_type == "regression" or not soft:
            pred_y = self.predictor.predict(sample, model="CatBoost")
        else:
            pred_y = self.predictor.predict_proba(sample, model="CatBoost")
        
        return pred_y


    def batch_synthetic(self, aug_size, soft=True):
        repeat = (aug_size) // self.data_df.shape[0] + 1
        tmp = pd.concat([self.data_df] * repeat)
        tmp_size = tmp.shape[0]
        rand_ind = self.rng.randint(0, tmp_size, size=tmp_size)
        other_tmp = tmp.iloc[rand_ind]
        coin_flip = self.rng.randint(0, 2, size=(tmp_size, self.col_count)).astype(bool)

        new_sample = tmp.where(coin_flip, other_tmp, axis=1)
        new_sample = new_sample.iloc[:aug_size]
        new_y = self._predict(new_sample, soft=soft)
        self.new_sample = new_sample
        self.new_y = new_y.values

        return new_sample, new_y


class DynaLMTabData(Dataset):
    def __init__(self, data_df, cat_embed, x_cat_encoder, header_embed, cat_dim_info, data_name, task_type, label_header, 
                 num_labels=None, embed_offset=True, y_encoder=None, header_names=None, augment=0., seed=0, 
                 synthetic_df=None, train_db=False) -> None:
        super().__init__()
        self.task_type = task_type

        if isinstance(data_df, tuple):
            data_df, y = data_df
        else:
            data_df, y = split_y_off(data_df, label_header)


        self._num_labels = num_labels

        self.Y = self.init_y(y, task_type, y_encoder=y_encoder)

        self.df_ind_map = self.init_df_order(data_df)

        if header_names is not None:
            num_headers, cat_headers = header_names
            self.cat_headers = cat_headers
            self.num_headers = num_headers


        if x_cat_encoder is not None:
            self.x_cat_encoder = x_cat_encoder
            self.cat_embed = torch.from_numpy(cat_embed.astype(np.float32))

            self.cat_dim_info = cat_dim_info
            cat_df = data_df.loc[:, data_df.dtypes == "string"]
            self.X_cat = self.init_cat_x(cat_df, x_cat_encoder, cat_dim_info, embed_offset=embed_offset)
            self.X_cat_vals, self.attn_mask = self.init_cat_cells(cat_embed, cat_dim_info)
        else:
            self.X_cat = None
            self.cat_embed = torch.empty((0, 10))

        
        num_df = data_df.loc[:, data_df.dtypes == "float32"]
        self.X_num = self.init_num_x(num_df)

        self.num_size = self.get_num_size()

        tmp = np.where(data_df.dtypes == "float32")[0]
        self.num_ind = tmp
        self.num_header_embed = torch.from_numpy(header_embed[tmp].astype(np.float32))
        tmp = np.where(data_df.dtypes != "float32")[0]
        self.cat_header_embed = torch.from_numpy(header_embed[tmp].astype(np.float32))

        self.name = data_name
        self.task_type = task_type

        self.augment = augment
        self.rng = np.random.RandomState(seed)

        self.synthetic_df = synthetic_df
        self.train_db = train_db


        if self.task_type != "regression":
            _, counts = np.unique(self.Y, return_counts=True)
            self.smooth_bias = counts / self.Y.shape[0]
        

    def init_df_order(self, data_df):
        cat_flags = data_df.dtypes == "string"
        cat_inds = []
        num_inds = []
        i = 0
        if self.task_type == "multiclass":
            cat_expand = len(np.unique(self.Y)) - 1
        else:
            cat_expand = 1
        for flag in cat_flags:
            if flag:
                for _ in range(cat_expand):
                    cat_inds.append(i)
                    i += 1
            else:
                num_inds.append(i)
                i += 1
        
        inds = cat_inds + num_inds
        self.cat_expand = cat_expand
        return np.asarray(inds)


    def init_y(self, y, task_type, y_encoder=None):
        if task_type != "regression":
            if y_encoder is None:
                y_encoder = preprocessing.LabelEncoder()
                Y = y_encoder.fit_transform(y.values).astype(np.int64)
            else:
                if len(y.shape) == 1:
                    Y = y_encoder.transform(y.values).astype(np.int64)
                else:
                    Y = y.values
            
        else:
            Y = y.values.astype(np.float32)
        self.y_encoder = y_encoder
        return Y

    
    def init_cat_x(self, cat_df, x_cat_encoder, cat_dim_info, embed_offset=True):
        cat_np = cat_df.values
        ind_offset = np.cumsum(cat_dim_info)
        self.vector_cat_dim = ind_offset[-1]
        ind_offset[1:] = ind_offset[:-1]
        ind_offset[0] = 0
        cat_X = x_cat_encoder.transform(cat_np)
        if embed_offset:
            cat_X += ind_offset
        self.cat_ind_offset = ind_offset
        return cat_X.astype(np.int64)
    

    def init_cat_cells(self, cat_embed, cat_dim_info):
        dim = cat_embed.shape[1]
        max_cat_len = max(cat_dim_info)
        cat_count = len(cat_dim_info)
        ind_offset = np.cumsum(cat_dim_info)
        ind_offset = np.insert(ind_offset, 0, 0)
        assert(ind_offset[-1] == cat_embed.shape[0])
        paras = torch.zeros((cat_count, max_cat_len, dim), dtype=torch.float)
        for i in range(cat_count):
            paras[i, 0:cat_dim_info[i]] = torch.from_numpy(cat_embed[ind_offset[i]:ind_offset[i+1]])
        
        attn_mask = torch.tensor(cat_dim_info).type(torch.IntTensor)
        
        return paras, attn_mask
    
    def init_num_x(self, num_df):
        if num_df.empty:
            return None
        return num_df.values
    

    def normalize_num(self, num_encoder=None):
        if self.X_num is not None and self.X_num.shape[0] > 0:
            if num_encoder is None:
                num_encoder = _get_normalizer(self.X_num, "standard")
                self.num_encoder = num_encoder
                self.X_num_normed = num_encoder.fit_transform(self.X_num)
            else:
                self.X_num_normed = num_encoder.transform(self.X_num)

        
    def get_num_size(self):
        if self.X_num is not None:
            return self.X_num.shape[1]
        return 0
    
    def get_cat_size(self):
        if self.X_cat is not None:
            return self.X_cat.shape[1]
        return 0


    def gen_y_normalizer(self, scheme):
        self.y_encoder = _get_normalizer(self.Y[:, None], scheme, noise=0.)
        self.Y = self.y_encoder.transform(self.Y[:, None])[:, 0]
        return self.y_encoder
    
    def set_y_normalizer(self, normalizer, transform=False):
        self.y_encoder = normalizer
        if transform:
            self.Y = self.y_encoder.transform(self.Y[:, None])[:, 0]


    def get_label_count(self):
        if self._num_labels is not None:
            return self._num_labels
        return len(np.unique(self.Y))
    
    def __len__(self):
        return self.Y.shape[0]


    def __getitem__(self, index) -> Any:
        coin_flip = self.rng.uniform()
        if self.X_cat is not None and self.X_num is not None:
            if coin_flip > self.augment:
                return [self.cat_embed, self.cat_header_embed, self.num_header_embed], [self.X_cat[index], self.X_num[index], self.X_num_normed[index]], self.label_smooth(self.Y[index])
            else:
                x, y = self.synthetic()
                return [self.cat_embed, self.cat_header_embed, self.num_header_embed], x, y
        
        if self.X_cat is not None:
            if coin_flip > self.augment:
                return [self.cat_embed, self.cat_header_embed, self.num_header_embed], [self.X_cat[index]], self.label_smooth(self.Y[index])
            else:
                x, y = self.synthetic()
                return [self.cat_embed, self.cat_header_embed, self.num_header_embed], x, y
        
        if self.X_num is not None:
            if coin_flip > self.augment:
                return [self.cat_embed, self.cat_header_embed, self.num_header_embed], [self.X_num[index], self.X_num_normed[index]], self.label_smooth(self.Y[index])
            else:
                x, y = self.synthetic()
                return [self.cat_embed, self.cat_header_embed, self.num_header_embed], x, y
    

    def label_smooth(self, y):
        return y 


    def synthetic(self):
        return self.synthetic_df.get()

    
    def init_dataloader(self, batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=False, worker_init_fn=init_worker_fn):
        self.data_loader = DataLoader(self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers,
                      pin_memory=pin_memory, worker_init_fn=worker_init_fn, collate_fn=custom_collate)
        
        self.batch_iter = self._batch_iterator()
    

    def shutdown_dataloader(self):
        self.batch_iter = None
        del self.data_loader
    
    def _batch_iterator(self):
        while True:
            for batch in self.data_loader:
                yield batch


    def get_cat_embed_size(self):
        return self.cat_embed.shape[0]


def _get_normalizer(data, scheme, seed=123, noise=1e-3):
    if scheme == 'minmax':
        normalizer = preprocessing.MinMaxScaler()
    elif scheme == "standard":
        normalizer = preprocessing.StandardScaler()
    elif scheme == 'quantile':
        normalizer = preprocessing.QuantileTransformer(
            output_distribution='uniform',
            n_quantiles=max(min(data.shape[0] // 30, 1000), 10),
            subsample=1000000000,
            random_state=seed,
        )
        if noise > 0:
            assert seed is not None
            stds = np.std(data, axis=0, keepdims=True)
            noise_std = noise / np.maximum(stds, noise)
            data = data + noise_std * np.random.default_rng(seed).standard_normal(data.shape)
    else:
        raise Exception("Uknown normalization cheme")
    normalizer.fit(data)
    return normalizer


@dataclass
class PeriodicOptions:
    n: int  # the output size is 2 * n
    sigma: float
    trainable: bool
    initialization: Literal['log-linear', 'normal']

def cos_sin(x: Tensor) -> Tensor:
    return torch.cat([torch.cos(x), torch.sin(x)], -1)


class PeriodicMultiBandwidth(nn.Module):
    def __init__(self, n_features: int, options: PeriodicOptions) -> None:
        super().__init__()
        self.n_features = n_features

        tmp_len = options.n // len(options.sigma)

        scale = torch.ones((1, options.n))
        
        for i, e in enumerate(options.sigma):
            scale[tmp_len*i:tmp_len*(i+1)] *= e


        ret = torch.normal(0.0, 1, (n_features, options.n)) * scale
        self.coefficients = nn.Parameter(ret)

        if not options.trainable:
            self.coefficients.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2
        assert x.shape[1] == self.n_features
        tmp =  cos_sin(self.coefficients[None] * x[:, None, ..., None])
        return tmp.mean(1)