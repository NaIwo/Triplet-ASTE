import json
import os
from os.path import join
from typing import Union, List, Dict

import fire
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from aste.configs import base_config
from aste.dataset.encoders import TransformerEncoder
from aste.dataset.reader import DatasetLoader
from aste.models import (
    BaseModel,
    SentimentPredictorTripletModel
)


def train_model(
        data_path: str = ".", dataset_name: str = '14lap', result_path: str = 'experiment_results',
        model_checkpoint_path: str = 'models/aste_model', experiment_id: int = 0,
):
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ CONFIG \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    config = base_config
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ DATA LOADER \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    model_setup = join('triplet_transformer', 'wo_amazon')
    model_checkpoint_path = join(model_checkpoint_path, model_setup, dataset_name, str(experiment_id))
    data_path = os.path.join(data_path, dataset_name)
    dataset_reader = DatasetLoader(data_path=data_path, encoder=TransformerEncoder(), config=config)

    train_data = dataset_reader.load('train.txt', num_workers=4, prefetch_factor=2, drop_last=True)
    dev_data = dataset_reader.load('dev.txt', shuffle=False, num_workers=4, prefetch_factor=2)
    test_data = dataset_reader.load('test.txt', shuffle=False, num_workers=4, prefetch_factor=2)

    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ MODEL \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # Here we create the model, you can use any model you want.
    model: BaseModel = SentimentPredictorTripletModel(config=config)

    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ TRAINER ELEMENTS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    # https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html#built-in-callbacks
    callbacks: List = [
        # EarlyStopping(monitor='val__final_metric_SpanF1', patience=12, verbose=True, mode='max'),
        ModelCheckpoint(dirpath=model_checkpoint_path, filename='aste_model', verbose=True,
                        monitor='val__final_metric_SpanF1', mode='max', every_n_epochs=1)
    ]

    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ W&B \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    wandb_logger = WandbLogger(project=f'Project', name=f'{dataset_name}_{experiment_id}')
    wandb_logger.experiment.config["batch_size"] = config['general-training']['batch-size']
    wandb_logger.experiment.config.update(config)
    wandb_logger.experiment.config["dataset_name"] = dataset_name
    wandb_logger.experiment.config["experiment_id"] = experiment_id

    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ TRAINER \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#basic-use
    # How much of training dataset to check (float = fraction, int = num_batches). Default: ``1.0``.
    limit_train_batches: Union[float, int] = 1.0  # for DEBUG (for example): set to 1 (int)
    limit_val_batches: Union[float, int] = 1.0
    limit_test_batches: Union[float, int] = 1.0

    trainer: Trainer = Trainer(
        logger=wandb_logger,
        enable_checkpointing=True,
        callbacks=callbacks,
        # accumulate_grad_batches=8,
        accelerator='gpu' if 'cuda' in config['general-training']['device'] else 'cpu',
        # devices=1,
        # strategy='deepspeed_stage_2',
        gradient_clip_val=0.8,
        min_epochs=30,
        max_epochs=130,
        max_time='00:12:00:00',  # DD:HH:MM:SS
        precision=config['general-training']['precision'],
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_data,
        val_dataloaders=dev_data
    )

    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ TEST \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    config['model']['remove-intersected'] = True
    model = SentimentPredictorTripletModel.load_from_checkpoint(
        checkpoint_path=trainer.checkpoint_callback.best_model_path, config=config).to(
        config['general-training']['device']
    )
    results = trainer.test(model=model, dataloaders=test_data)

    model = SentimentPredictorTripletModel.load_from_checkpoint(
        checkpoint_path=trainer.checkpoint_callback.best_model_path, config=config).to(
        config['general-training']['device']
    )

    pred_results = model.predict(test_data)
    t_results = [r.outputs['final_triplet'].to_string() for r in pred_results]
    path = os.path.join(result_path, dataset_name, model_setup, f'{dataset_name}_{experiment_id}_pred.txt')
    save_list_to_file(t_results, path)

    path = os.path.join(result_path, dataset_name, model_setup, f'results_{experiment_id}.json')
    save_results(results, path)

    # Precision recall lines
    config['general-training']['batch-size'] = 1
    dev_data = dataset_reader.load('dev.txt', shuffle=False, num_workers=1, prefetch_factor=1)
    model.triplet_precision_recall_different_thresholds(dev_data)

    del model


def save_results(results: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for result in results:
            result = to_float(result)
            json.dump(result, f)


def save_list_to_file(data: List[str], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for line in data:
            if line == '':
                continue
            f.write(line + "\n")


def to_float(data: Dict) -> Dict:
    for key, value in data.items():
        if isinstance(value, dict):
            data[key] = to_float(value)
        else:
            data[key] = float(value)
    return data


if __name__ == '__main__':
    fire.Fire(train_model)
