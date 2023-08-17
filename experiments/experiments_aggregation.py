import os
import json
from collections import defaultdict
from typing import Dict, DefaultDict

from aste.utils import to_json

INTERNAL_PATH: str = os.path.join('transformer', 'food_reviews_longer')


def aggregate(path: str) -> None:
    num_of_metric_files: int = 0
    results: DefaultDict = defaultdict(float)
    file_name: str
    for file_name in os.listdir(path):
        if file_name.startswith('results_'):
            num_of_metric_files += 1
            file_path = os.path.join(path, file_name)
            with open(file_path) as json_file:
                result: Dict = json.load(json_file)

            for key, value in result.items():
                results[key] += value

    for key, value in results.items():
        results[key] = value / num_of_metric_files

    to_json(results, os.path.join(path, 'final_results.json'), mode='w')

    print(results)


if __name__ == '__main__':
    dataset_name: str
    for dataset_name in ['14res']:
        print(dataset_name)
        data_path: str = os.path.join('experiments_results_newest', dataset_name, INTERNAL_PATH)
        aggregate(data_path)
