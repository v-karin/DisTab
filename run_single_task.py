from autogluon.tabular import TabularPredictor
import numpy as np
import torch
import torch.nn as nn
from torch import optim, Tensor
from torchmetrics import Metric
import torchmetrics
import torch.nn.functional as F
from pytorch_lightning.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
from lightning import seed_everything

import os
from glob import glob
import shutil
import gin
import argparse
import warnings

def get_parser():
    parser = argparse.ArgumentParser(description='Test DisTab.')

    parser.add_argument('--gin_file', type=str, default='gin_config/single_task.gin')
    parser.add_argument('--task_name', type=str, default='adult')
    parser.add_argument('--active_teacher_model', action="store_true",
                            help = 'Whether active teacher model training.')
    parser.add_argument('--active_pre_training', action="store_true",
                            help = 'Whether active DisTab pre-training.')
    parser.add_argument('--active_fine_tuning', action="store_true",
                            help = 'Whether active DisTab fine-tuning.')
    return parser

''' 
Tree-based model
'''

hyper_model_map = {
    'RF': "RandomForest",   # RandomForest (random forest)
    'XGB': "XGBoost",  # XGBoost
    'GBM': "LightGBM",  # LightGBM/LGBM
    'CAT': "CatBoost",  # CatBoost
    'FASTAI': "NeuralNetFastAI",   # NeuralNetFastAI (neural network with FastAI backend)
    'NN_TORCH': "NeuralNetTorch",

}

def convert_tp_save_format(task_ret_map, models):
    full_rets = []
    task_names = list(task_ret_map.keys())
    for task in task_names:
        full_rets.append(task_ret_map[task])

    full_rets = np.asarray(full_rets) # task x model_name x folds

    return {
        "task_names": task_names,
        "models": models,
        "ret": full_rets
}
    
def save_tree_result(task_name, task_type, eval_metric, models, folds_ret):
    # collect results of all tested tasks
    tree_ret_map = {}
    
    tree_ret_map[task_name] = np.asarray(folds_ret).transpose(1, 0)
    
    save_file = ""
    if task_type == "regression":
        save_file = "baseline_tree_res.res"
    elif task_type == "binary":
        save_file = "baseline_tree_binary_%s.res" % eval_metric
    elif task_type == "multiclass":
        save_file = "baseline_tree_multi_%s.res" % eval_metric
    else:
        raise ValueError("Invalid task_type provided.")
    
    save_tree = convert_tp_save_format(tree_ret_map, models)

    torch.save(save_tree, save_file)

@gin.configurable
def run_teacher_model(run_config, task_name, task_type, fold, eval_metric, label_header, train_data, test_data):
    print('Running teacher model generation ...')
    
    model_hypers = {}
    if run_config["hyperparameters"] is None:
        for model in run_config["models"]:
            model_hypers[model] = {}
    
    predictor = TabularPredictor(label=label_header,
                                            path = f"{run_config['save_dir']}/{task_name}_{fold}_{eval_metric}/",
                                            problem_type = task_type,
                                            eval_metric = eval_metric, verbosity = run_config["verbose"],
                                            log_to_file = False).fit(train_data,
                                                                    hyperparameters = model_hypers,
                                                                    auto_stack = False,
                                                                    num_bag_folds=0,
                                                                    num_stack_levels=0,
                                                                    fit_weighted_ensemble = False,
                                                                    )
        
    models_ret = []
    for model in run_config["models"]:
        tmp = predictor.evaluate(test_data, model = hyper_model_map[model])[eval_metric]
        models_ret.append(tmp)
    
    return run_config["models"], models_ret

''' 
DisTab model
'''

class LogLoss(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("logprob", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        log_loss = F.cross_entropy(preds, target, reduction="sum")

        self.logprob += log_loss
        self.total += target.numel()

    def compute(self) -> Tensor:
        return self.logprob / self.total

def auc_transform(y_pred):
    y_pred = F.softmax(y_pred, dim=-1)[:, 1]
    return y_pred

def acc_transform(y_pred):
    y_pred = torch.argmax(y_pred, dim=-1)
    return y_pred

class LModule(L.LightningModule):
    metric_transform = {
        "auc": auc_transform,
        "acc": acc_transform,
        "logloss": torch.nn.Identity(),
        "rmse": torch.nn.Identity()
    }
    def __init__(self, model, lr, task_type, num_labels, test_names, batch_size):
        super().__init__()
        self.model = model

        self.lr = lr
        self.batch_size = batch_size
        self.log_keys = []
        self.metrics = nn.ModuleList()
        if task_type == "regression":
            metric_names = {
                "rmse": lambda : torchmetrics.regression.MeanSquaredError(False)
            }
            self.loss_fn = torch.nn.MSELoss()
        elif task_type == "binary":
            metric_names = {
                "auc": lambda: torchmetrics.classification.AUROC(task="binary"),
                "acc": lambda : torchmetrics.classification.Accuracy(task="binary")
            }
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif task_type == "multiclass":
            metric_names = {
                "logloss": LogLoss,
                "acc": lambda : torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_labels)
            }
            self.loss_fn = torch.nn.CrossEntropyLoss()
        
        self.metric_count = len(metric_names)

        for _ in test_names:
            for name, metric_fn in metric_names.items():
                self.log_keys.append(f"{name}")
                self.metrics.append(metric_fn())


    def training_step(self, batch, batch_idx):
        header_info, x, y = batch
        pred = self.model(*header_info, x).squeeze()
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=self.batch_size)

        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        header_info, x, y = batch
        pred = self.model(*header_info, x).squeeze(dim=-1)

        for i in range(self.metric_count * dataloader_idx, self.metric_count * (dataloader_idx + 1)):
            tmp = self.metric_transform[self.log_keys[i]](pred)
            self.metrics[i](tmp, y)
            self.log(self.log_keys[i], self.metrics[i], on_step=False, on_epoch=True)

    def set_optimizer(self, optim, sch=None):
        self.optimizer = optim
        self.sch = sch

    def configure_optimizers(self):
        if self.sch is None:
            return self.optimizer
        return [self.optimizer], [self.sch]

from dyna_tab_model import build_model
from data import get_data_splits, get_synthetic_data_splits

import pandas as pd
def get_results(log_root, task_type):
    metrics = f"{log_root}/metrics.csv"
    metrics = pd.read_csv(metrics)
    metrics = metrics.iloc[::2]
    contexts = ["dataloader_idx_0", "dataloader_idx_1"]
    if task_type == "regression":
        val_cols = ["rmse"]
        signs = [-1]
    elif task_type == "binary":
        val_cols = ["auc", "acc"]
        signs = [1, 1]
    else:
        val_cols = ["logloss", "acc"]
        signs = [-1, 1]
    
    ret = {}
    
    for key, sign in zip(val_cols, signs):
        tmp = []
        for e in contexts:
            tmp.append(f"{key}/{e}")
        fragment = metrics[tmp].values * sign
        ret[key] = fragment
    
    return ret

def model_soup_by_name(checkpoints):
    save_dir = os.path.dirname(checkpoints[0])
    metric_type = os.path.basename(checkpoints[0]).split("-")[0]
    base = torch.load(checkpoints[0])
    state_dict = base["state_dict"]
    for t_path in checkpoints[1:]:
        t_state = torch.load(t_path)["state_dict"]
        for key in state_dict:
            state_dict[key] = state_dict[key] + t_state[key]
            
    for key in state_dict:
        state_dict[key] = state_dict[key] / len(checkpoints)

    base["state_dict"] = state_dict
    torch.save(base, f"{save_dir}/{metric_type}_soup.ckpt")
    
def model_soup(log_root, metric_type):
    checkpoints = glob(f"{log_root}/{metric_type}*.ckpt")
    model_soup_by_name(checkpoints)

def run_test(trainer, wrapper, data, checkpoint):
    _, test_data = data
    tmp = torch.load(checkpoint)
    wrapper.load_state_dict(tmp["state_dict"], strict=False)
    return trainer.validate(model = wrapper, dataloaders = test_data.data_loader)[0]

def what_to_merge(distill_res, finetune_res, save_dir, metric_name):
    finetune_res = finetune_res[metric_name][:, 0]
    if distill_res * finetune_res[0] < 0:
        distill_res *= -1
    
    checkpoints = glob(f"{save_dir}/{metric_name}-epoch*.ckpt")
    top_res = {}
    for e in checkpoints:
        file_name = os.path.basename(e)
        segs = file_name.split("-")
        epoch_num = int(segs[1][6:-5])
        top_res[e] = finetune_res[epoch_num]
    good_merge = [e for e, res in top_res.items() if res > distill_res ]
    return good_merge

def test_main(trainer, wrapper, data, metric_type, log_root, scaling=1., checkpoints=None):
    if checkpoints is None:
        checkpoints = [f"{log_root}/{metric_type}_soup.ckpt"]
    ret = {}
    wrapper.eval()
    for e in checkpoints:
        ret[e] = run_test(trainer, wrapper, data, e)[metric_type] * scaling
    wrapper.train()
    return ret

def new_trainer(run_config, cbs, train_data):
    gpu = "gpu" if torch.cuda.is_available() else "cpu"
    devices = run_config["devices"] if (gpu == "gpu") else 1
    return L.Trainer(max_epochs=run_config["epoch"],
                     callbacks= cbs,
                     logger=False,
                     accelerator=gpu,
                     devices=devices,
                     num_nodes=1,
                     deterministic=True,
                     log_every_n_steps=min(50, len(train_data.data_loader))
                     )

@gin.configurable
def distab_model(arch_config, run_config, task_type_metric, task_name, task_type, task, tab_data, fold):
    print("Run distab_model...") 
    ea_metric = task_type_metric[task_type]
    if run_config["synthetic"]:
        train_data, val_data, test_data = get_synthetic_data_splits(model_dir=run_config["teacher_model_dir"], data_name=task_name,
                                                    task=task, tab_data=tab_data, fold=fold, pct=0.1, seed=0, y_transform=True)
    else:
        train_data, val_data, test_data = get_data_splits(task_name, task, tab_data, fold=fold, pct=0.1, seed=0,
                                                    augment=run_config["augment"], y_transform=True)
        
    cand_train_size = [1024, 512, 256, 128]
    train_size = len(train_data)
    for batch_size in cand_train_size:
        if train_size / batch_size > 30:
            break
    
    train_data.init_dataloader(batch_size, shuffle=True, drop_last=True, worker_init_fn=None)
    val_data.init_dataloader(run_config["test_bsz"], shuffle=False, drop_last=False, worker_init_fn=None)
    test_data.init_dataloader(run_config["test_bsz"], shuffle=False, drop_last=False, worker_init_fn=None)
    
    cat_embed_size = train_data.get_cat_embed_size()
    
    if train_data.task_type == "regression":
        out_dim = 1
        scaling = np.sqrt(train_data.y_encoder.var_)
        metric_names = [("rmse", "min")]
    else:
        out_dim = train_data.get_label_count()
        scaling = 1.
        if task_type == "binary":
            metric_names = [("auc", "max"), ("acc", "max")]
        else:
            metric_names = [("logloss", "min"), ("acc", "max")]
        
    tab_model = build_model(**arch_config, num_ind=train_data.num_ind, cat_size=train_data.get_cat_size(),
                            out_dim=out_dim, cat_embed_size=cat_embed_size)

    tab_in_dim = arch_config["tab_in_dim"]
    raw_tokens = arch_config["lm_tokens"]
    exp_name = f"{task_name}_{tab_in_dim}_{raw_tokens}_{fold}"

    csv_logger = CSVLogger(run_config["log_root"], exp_name)
    save_dir = csv_logger.log_dir

    csv_logger.log_hyperparams(arch_config | run_config)

    cbs = []
    for name, mode in metric_names:
        monitor = f"{name}/dataloader_idx_0"
        tmp = ModelCheckpoint(dirpath=save_dir, save_top_k=run_config["top_k"], monitor=monitor, filename=f"{name}-{{epoch}}", mode=mode)
        cbs.append(tmp)

    trainer = new_trainer(run_config, cbs, train_data)
    wrapper = LModule(tab_model, run_config["lr"], task_type, num_labels=train_data.get_label_count(), test_names=["val", "test"], batch_size=batch_size)

    parameters = tab_model.parameters()
    weight_decay = run_config["weight_decay"]

    if run_config["pre_trained"]:
        res_dict = torch.load(run_config["pre_trained"])
        if fold >= len(res_dict[task_name]["log_dir"]):
            raise ValueError(
                f"The current fold of fine-tuning is {fold}, which is out of range of pre-training folds. "
                "Please re-run the pre-training for the task with the required number of folds."
                )
        checkpoint = res_dict[task_name]["log_dir"][fold]
        pre_checkpoint = f"{os.path.dirname(checkpoint)}/{metric_names[ea_metric][0]}_soup.ckpt"
        tmp = torch.load(checkpoint)
        wrapper.load_state_dict(tmp["state_dict"])

        trainer.logger = False
        distill_res = {}
        for name, _ in metric_names:
            t_tmp = test_main(trainer, wrapper, (test_data, val_data), name, os.path.dirname(checkpoint), checkpoints=[pre_checkpoint])
            distill_res[name] = t_tmp[pre_checkpoint]

        for name, _ in metric_names:
            print(f"pre-trained perf is {res_dict[task_name][name][fold]}")
            
    optimizer = optim.AdamW(
        parameters,
        lr=run_config["lr"],
        weight_decay=weight_decay
    )

    sch = None
    
    wrapper.set_optimizer(optimizer, sch)
    trainer.logger = csv_logger
    trainer.fit(wrapper, train_data.data_loader, [val_data.data_loader, test_data.data_loader])
    
    if run_config["pre_trained"]:
        finetune_res = get_results(save_dir, task_type)
        ret = {}
        for name, _ in metric_names:
            merge_checkpoints = what_to_merge(distill_res[name], finetune_res, save_dir, name)
            if len(merge_checkpoints) < run_config["top_k"]:
                copied_target = f"{save_dir}/{name}-epoch=n1.ckpt"
                shutil.copyfile(pre_checkpoint, copied_target)
                merge_checkpoints.append(copied_target)
            model_soup_by_name(merge_checkpoints)
            ret[name] = test_main(trainer, wrapper, (val_data, test_data), name, save_dir)
    else:
        ret = {}
        trainer.logger = False
        for name, _ in metric_names:
            model_soup(save_dir, name)
            ret[name] = test_main(trainer, wrapper, (val_data, test_data), name, save_dir)
    
    train_data.shutdown_dataloader()
    val_data.shutdown_dataloader()
    test_data.shutdown_dataloader()
    
    return ret, scaling

def save_distab_result(task_name, folds_ret, folds_scale, ret, scaling, save_path):
    # collect results of all tested tasks
    rets = {}
    
    if os.path.exists(save_path):
        rets = torch.load(save_path)
    
    folds_ret.append(ret)
    folds_scale.append(scaling)
    folds_ret_scale = post_process_fold_ret(folds_ret)
    folds_ret_scale["scale"] = np.asarray(folds_scale).squeeze()
        
    rets[task_name] = folds_ret_scale
    torch.save(rets, save_path)

def load_distab_result(task_type):
    task_name = gin.query_parameter('run_task.task_name')
    active_distab = gin.query_parameter('run_fold.run_config')['run_DisTab']
    active_fine_tuning = gin.query_parameter('run_distab.fine_tuning_config')['activate_fine_tuning']
    ret_file = gin.query_parameter('run_distab.fine_tuning_config')['save_path']
    if isinstance(ret_file, gin.config.ConfigurableReference):
        ret_file = gin.query_parameter(str(ret_file))  # Resolve the macro

    metric_map = gin.query_parameter('run_fold.run_config')['distab_type_metric']

    metric = ''
    ret = []
    ave = 0
    if active_distab and active_fine_tuning:
        raw_results = torch.load(ret_file)
        metric = list(raw_results[task_name].keys())[metric_map[task_type]]
        ret = raw_results[task_name][metric]
        scale = raw_results[task_name]['scale']
        if len(ret) == 5:
            ave = np.mean(ret * scale)
        else:
            warnings.warn("5 fold is not satisfied", category=UserWarning)

    print(f'The average performance of task {task_name} with metric {metric} on {len(ret)} is: {ave}')

@gin.configurable
def run_distab(pre_training_config, fine_tuning_config, task_name, task_type, task, tab_data, fold, task_type_metric, \
    folds_ret_pre_training, folds_scale_pre_training, folds_ret_fine_tuning, folds_scale_fine_tuning):
    if pre_training_config["activate_pre_training"]:
        print('Running pre-training ...')
        seed_everything(0, workers=True)
        ret, scaling = distab_model(run_config=pre_training_config, task_type_metric=task_type_metric, task_name=task_name, task_type=task_type, task=task, tab_data=tab_data, fold=fold)
        save_distab_result(task_name, folds_ret_pre_training, folds_scale_pre_training, ret, scaling, pre_training_config["save_path"])
        
    if fine_tuning_config["activate_fine_tuning"]:
        print('Running fine-tuning ...')
        seed_everything(0, workers=True)  
        ret, scaling = distab_model(run_config=fine_tuning_config, task_type_metric=task_type_metric, task_name=task_name, task_type=task_type, task=task, tab_data=tab_data, fold=fold)
        save_distab_result(task_name, folds_ret_fine_tuning, folds_scale_fine_tuning, ret, scaling, fine_tuning_config["save_path"])

from data import get_openml_task, get_tab_data
from config import openml_data_path as data_root
  
@gin.configurable
def run_fold(run_config, task_name, fold, folds_ret_pre_training, folds_scale_pre_training, folds_ret_fine_tuning, folds_scale_fine_tuning):

    task = get_openml_task(task_name)
    tab_data = get_tab_data(task_name, data_root=data_root)
    df, _, _, task_type, label_header = tab_data["df_data"]
    # df: DataFrame, task_type: the type of task, label_header: the header of lable.

    tree_eval_metric = run_config["tree_type_metric"][task_type]

    teacher_models = []
    teacher_models_ret = {}
    if run_config["run_teacher_model"]:
        '''Prepare data for tree-based teacher model'''
        train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=fold)
        train_data = df.iloc[train_indices]
        test_data = df.iloc[test_indices]

        teacher_models, teacher_models_ret = run_teacher_model(task_name=task_name, task_type=task_type, fold=fold, eval_metric=tree_eval_metric, label_header=label_header, train_data=train_data, test_data=test_data)

    if run_config["run_DisTab"]:
        run_distab(task_name=task_name, task_type=task_type, task=task, tab_data=tab_data, fold=fold, task_type_metric=run_config["distab_type_metric"],
                folds_ret_pre_training=folds_ret_pre_training, folds_scale_pre_training=folds_scale_pre_training, \
                folds_ret_fine_tuning=folds_ret_fine_tuning, folds_scale_fine_tuning=folds_scale_fine_tuning)

    return teacher_models, teacher_models_ret, task_type, tree_eval_metric

def post_process_fold_ret(folds_ret):
    metrics = list(folds_ret[0].keys())
    ret = {}
    for metric in metrics:
        t_log_dir = []
        t_eval_res = []
        for fold_ret in folds_ret:
            t_ret = fold_ret[metric]
            t_log_dir.append(list(t_ret.keys())[0])
            t_eval_res.append(list(t_ret.values())[0])

        ret[metric] = np.asarray(t_eval_res)
    ret["log_dir"] = t_log_dir
    return ret

@gin.configurable
def run_task(task_name, folds):
    torch.set_float32_matmul_precision('medium')
    print(f"================== task: {task_name} ==================")
    
    folds_ret_teacher = []
    folds_ret_pre_training = []
    folds_scale_pre_training = []
    folds_ret_fine_tuning = []
    folds_scale_fine_tuning = []
    task_type = ''
    for fold in range(folds):
        print(f"------------ fold: {fold} -----------")
        teacher_models, teacher_models_ret, task_type, eval_metric \
        = run_fold(task_name=task_name, fold=fold, folds_ret_pre_training=folds_ret_pre_training, \
            folds_scale_pre_training=folds_scale_pre_training, folds_ret_fine_tuning=folds_ret_fine_tuning, \
                folds_scale_fine_tuning=folds_scale_fine_tuning)

        if teacher_models_ret:
            folds_ret_teacher.append(teacher_models_ret)
    
    load_distab_result(task_type)

    if folds_ret_teacher:
        save_tree_result(task_name, task_type, eval_metric, teacher_models, folds_ret_teacher)

def bind_nested_dict_param(gin_key_path, value):
    keys = gin_key_path.split(".")
    base_key = ".".join(keys[:-1])
    nested_key = keys[-1]
    base_dict = gin.query_parameter(base_key)
    base_dict[nested_key] = value
    gin.bind_parameter(base_key, base_dict)

def override_gin_param(args):
    if args.task_name is not None:
        gin.bind_parameter("run_task.task_name", args.task_name)
    if not args.active_teacher_model:
        bind_nested_dict_param('run_fold.run_config.run_teacher_model', args.active_teacher_model)
    if not args.active_pre_training:
        bind_nested_dict_param('run_distab.pre_training_config.activate_pre_training', args.active_pre_training)
    if not args.active_fine_tuning:
        bind_nested_dict_param('run_distab.fine_tuning_config.activate_fine_tuning', args.active_fine_tuning)

def run_task_by_gin(args):
    gin.parse_config_file(args.gin_file)
    override_gin_param(args)

    run_task()

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    run_task_by_gin(args)