import json
import os
from typing import Union, List, Dict

import fire
from aste.configs import base_config
from aste.dataset.encoders import TransformerEncoder
from aste.dataset.reader import DatasetLoader
from aste.models import BaseModel, TripletModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def train_model(data_path: str = "/home/iwo/Pulpit/Studia/Triplet/ASTE/dataset/data/ASTE_data_v2",
                dataset_name: str = '14lap',
                result_path: str = 'experiments/experiment_results',
                model_checkpoint_path: str = 'models/aste_model',
                experiment_id: int = 0,
                ):
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ CONFIG \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # config = setup_config(path='.')
    config = base_config
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ DATA LOADER \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    data_path = os.path.join(data_path, dataset_name)
    dataset_reader = DatasetLoader(data_path=data_path, encoder=TransformerEncoder(), config=config)

    train_data = dataset_reader.load('train.txt')
    dev_data = dataset_reader.load('dev.txt', shuffle=False)
    test_data = dataset_reader.load('test.txt', shuffle=False)

    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ MODEL \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    model: BaseModel = TripletModel(config=config)

    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ TRAINER ELEMENTS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    # https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html#built-in-callbacks
    callbacks: List = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=True, mode='min'),
        ModelCheckpoint(dirpath=model_checkpoint_path, filename='aste_model', verbose=True,
                        monitor='val_loss', save_last=True, mode='min', every_n_epochs=2)
    ]

    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ W&B \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    wandb_logger = WandbLogger(project='triplet-sent')
    wandb_logger.experiment.config["batch_size"] = config['general-training']['batch-size']
    wandb_logger.experiment.config.update(config)
    wandb_logger.experiment.config["dataset_name"] = dataset_name
    wandb_logger.experiment.config["experiment_id"] = experiment_id

    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ TRAINER \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#basic-use
    # How much of training dataset to check (float = fraction, int = num_batches). Default: ``1.0``.
    limit_train_batches: Union[float, int] = 10  # for DEBUG (for example): set to 1 (int)
    limit_val_batches: Union[float, int] = 10
    limit_test_batches: Union[float, int] = 1.0

    trainer: Trainer = Trainer(
        logger=wandb_logger,
        # enable_checkpointing=True,
        # callbacks=callbacks,
        accumulate_grad_batches=16,
        accelerator=config['general-training']['device'],
        devices=1,
        strategy='ddp',
        min_epochs=1,
        max_epochs=120,
        max_time='00:12:00:00',  # DD:HH:MM:SS
        precision=config['general-training']['precision'],
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_data,
        val_dataloaders=train_data
    )

    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ TEST \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    results = trainer.test(model=model, dataloaders=test_data, ckpt_path='best')
    result_path = os.path.join(result_path, dataset_name, f'results_{experiment_id}.json')
    save_results(results, result_path)
    del model


def save_results(results: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        for result in results:
            result = to_float(result)
            json.dump(result, f)


def to_float(data: Dict) -> Dict:
    for key, value in data.items():
        if isinstance(value, dict):
            data[key] = to_float(value)
        else:
            data[key] = float(value)
    return data


if __name__ == '__main__':
    fire.Fire(train_model)
