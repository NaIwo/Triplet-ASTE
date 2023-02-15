from aste.dataset.reader import DatasetLoader
from aste.configs import config , set_up_logger
from aste.trainer import Trainer
from aste.models import BaseModel, TransformerBasedModel
from aste.dataset.encoders import TransformerEncoder

import logging
from typing import Dict
import os




def log_introductory_info(data_path: str) -> None:
    logging.info(f"Data path: {data_path}")
    logging.info(f"Batch size: {config['general-training']['batch-size']}")
    logging.info(f"Effective batch size: {config['dataset']['effective-batch-size']}")


if __name__ == '__main__':
    dataset_name: str = '14lap'
    data_path: str = os.path.join(os.getcwd(), 'dataset', 'data', 'ASTE_data_v2', dataset_name)

    set_up_logger()

    dataset_reader = DatasetLoader(data_path=data_path, encoder=TransformerEncoder(),
                                   include_sub_words_info_in_mask=False)

    # train_data = dataset_reader.load('train.txt')
    # dev_data = dataset_reader.load('dev.txt')
    test_data = dataset_reader.load('test.txt')

    model: BaseModel = TransformerBasedModel()

    tracker: BaseTracker = BaseTracker(project="...", entity="...")

    log_introductory_info(data_path)

    save_path: str = os.path.join(os.getcwd(), 'results', 'chunk_simple', 'model.pth')
    trainer: Trainer = Trainer(model=model, tracker=tracker, save_path=save_path)

    trainer.train(train_data=test_data, dev_data=test_data)

    # trainer.load_model(save_path)

    # trainer.check_coverage_detected_spans(test_data)
    # results: Dict = trainer.test(test_data)

    local_results: Dict = trainer.test(test_data)
    # coverage_results: Dict = trainer.check_coverage_detected_spans(test_data)
    #
    # coverage_save_path: str = os.path.join(os.getcwd(), 'experiments', 'experiment_results', 'endpoint',
    #                                        f'{dataset_name}', 'all_spans_creator', f'coverage_results_{0}.json')
    # to_json(data_to_save=coverage_results, path=coverage_save_path)
    #
    # metric_save_path: str = os.path.join(os.getcwd(), 'experiments', 'experiment_results', 'endpoint', f'{dataset_name}',
    #                                      'all_spans_creator', f'metrics_results_{0}.json')
    # local_results[ModelMetric.NAME].to_json(path=metric_save_path)
    # results: List[ModelOutput] = trainer.predict(test_data)
    # save_path: str = os.path.join(os.getcwd(), 'results', '14lap_res.txt')
    # ModelOutput.save_list_of_outputs(results, save_path)

    # sentence = Sentence('i hate this screen ')
    # prediction: ModelOutput = trainer.predict(sentence)
    # print(prediction)
