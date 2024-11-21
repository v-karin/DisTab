<div align="center">

# Deep Tabular Learning via Distillation and Language Guidance

[**Overview**](#overview)
| [**Requirements**](#requirements)
| [**Datasets**](#datasets)
| [**Running DisTab**](#running-distab)
<!-- | [**Citation**](#citation) -->

</div>

## Overview
This is the official implementation for [**Deep Tabular Learning via Distillation and Language Guidance**](https://openreview.net/pdf?id=p6KIteShzf) (DisTab). DisTab is based on transformer architectures, and leverages distillation pre-training and language-guided embeddings for robust peformance. The repository provides sample code and usage guide.


## Requirements
Key dependencies include PyTorch, PyTorch Lightning, OpenML, AutoGluon, gin-config, scikit-learn, and pandas.

## Datasets
We preprocess several [datasets](https://drive.google.com/file/d/1P26lMRBLFpbgmTlXoAelGvtqQcfJn045/view?usp=sharing) from OpenML for running DisTab. The pre-processed datasets include the language-guided embeddings as described in the paper, using [Llama-3-8B](https://www.llama.com/llama-downloads/) as the embedding model. To use the datasets, download and unzip them under the repo. `dataset` folder should be created and origanized as follows:

```
dataset
├── adult
│   ├── head.json
│   ├── tab_data
├── higgs
│   ├── head.json
│   ├── tab_data
├── ...
```

Each dataset includes `head.json` (metadata, e.g. OpenML Task ID) and `tab_data` (the preprocessed tabular data).

## Running DisTab

### Configuration
Please configure experiment settings and model hyperparameters in gin-config files located in the `gin_config` folder.

The folder structure of gin-config:
```
gin_config
├── single_task.gin
├── ...
```

### Running

**DisTab** consists of three components, including *training a teacher model (tree-based models)*, *pre-training by distilling the teacher model*, and *model fine-tuning*. Each stage may be run independently for convenience.

For full training including the training of a teacher model, distillation pre-training and fine-tuning:
```bash
python run_single_task.py --gin_file gin_config/single_task.gin --task_name adult  --active_teacher_model --active_pre_training --active_fine_tuning
```
The teacher models are saved in `teacher_model_dir` from `single_task.gin`, and the performance result in `baseline_tree_<task_type>_<metric>.res`. The fine-tuned models are saved in `fine_tuned_model_dir` from `single_task.gin`, with the performance results in `fine_tuned_result_path` from `single_task.gin`.

To only train the teacher model:
```bash
python run_single_task.py --gin_file gin_config/single_task.gin --task_name {task_name}  --active_teacher_model
```

To only train DisTab if a teacher model is available:
```bash
python run_single_task.py --gin_file gin_config/single_task.gin --task_name {task_name} --active_pre_training --active_fine_tuning
```

<!-- ## Citation
If you find [**DisTab**](https://openreview.net/pdf?id=p6KIteShzf) helpful in your research, please cite the following paper:
```bibtex
@article{wang2024distab,
  title={Deep Tabular Learning via Distillation and Language Guidance},
  author={Ruohan Wang, Wenhao Fu, Carlo Ciliberto},
  journal={Transactions on Machine Learning Research},
  year={2024}
}
``` -->
