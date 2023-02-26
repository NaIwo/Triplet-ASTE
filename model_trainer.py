import os
from typing import Union, List

import fire
from aste.configs import base_config, setup_config
from aste.dataset.encoders import TransformerEncoder
from aste.dataset.reader import DatasetLoader
from aste.models import BaseModel, TripletModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def train_model(data_path: str = os.path.join('.', 'dataset', 'data', 'ASTE_data_v2'),
                dataset_name: str = '14lap',
                api_key: str = 'ANONYMOUS',
                model_checkpoint_path: str = '../models/aste_model'
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
                        monitor='val_loss', save_last=True, mode='min')
    ]

    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ TRAINER \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#basic-use
    # How much of training dataset to check (float = fraction, int = num_batches). Default: ``1.0``.
    limit_train_batches: Union[float, int] = 1.0  # for DEBUG (for example): set to 1 (int)
    limit_val_batches: Union[float, int] = 1.0
    limit_test_batches: Union[float, int] = 1.0

    trainer: Trainer = Trainer(
        # logger=logger,
        # enable_checkpointing=True,
        # callbacks=callbacks,
        accumulate_grad_batches=16,
        accelerator=config['general-training']['device'],
        devices=1,
        strategy='ddp',
        min_epochs=1,
        max_epochs=120,
        max_time='00:12:00:00', # DD:HH:MM:SS
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


if __name__ == '__main__':
    # set_up_logger()
    fire.Fire(train_model)
