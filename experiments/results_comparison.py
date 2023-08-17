import json
from collections import defaultdict
from os import walk
from os.path import join, exists, relpath
from typing import List, Dict, DefaultDict

import pandas as pd
from aste.utils import to_json

from other_approach_results import OTHER_RESULTS

datasets: List = ['14lap', '14res', '15res', '16res']

score_results_file_name: str = 'final_results.json'
results_path: str = join('experiments_results_newest')


def process(results_file_name, add_other_results: bool = False) -> Dict:
    results: Dict = dict()
    dataset: str
    for dataset in datasets:
        path: str = join(results_path, dataset)
        results[dataset] = process_single_dataset(path, results_file_name)
        if add_other_results:
            add_other_results_f(results, dataset)

    return results


def process_single_dataset(data_path: str, results_file_name: str) -> Dict:
    results: Dict = dict()

    for root, dirs, files in walk(data_path):
        for dir_name in dirs:
            res_path: str = join(root, dir_name, results_file_name)
            if not exists(res_path):
                continue
            with open(res_path, 'r') as f:
                rel_dir = relpath(root, data_path)
                dir_name = join(rel_dir, dir_name)
                results[dir_name] = to_percent(json.load(f))
    return results


def to_percent(results: Dict) -> Dict:
    return {k.replace('test__final_metric_Span', ''): round(v * 100, 2) for k, v in results.items() if 'final_metric' in k}


def add_other_results_f(results: Dict, dataset_name: str) -> None:
    other_name: str
    for other_name in OTHER_RESULTS.keys():
        if dataset_name not in OTHER_RESULTS[other_name]:
            continue
        results[dataset_name][other_name] = OTHER_RESULTS[other_name][dataset_name]


def results_as_pandas(results, orient: str = 'index') -> pd.DataFrame:
    pd_results = pd.DataFrame.from_dict({(i, j): results[i][j]
                                         for i in results.keys()
                                         for j in results[i].keys()},
                                        orient=orient, dtype='float')
    return pd_results


def make_flatten_pd(results: pd.DataFrame) -> pd.DataFrame:
    results_dict: DefaultDict = results.to_dict()
    flatten_results = defaultdict(dict)
    dataset_name: str
    model_name: str
    score: float
    for (dataset_name, model_name), score in results_dict.items():
        flatten_results[dataset_name].update({model_name: score})

    return pd.DataFrame.from_dict(flatten_results).T


if __name__ == '__main__':
    # Scores as Dict
    scores: Dict = process(score_results_file_name, add_other_results=True)

    print(json.dumps(scores, indent=2))

    # Scores as json
    to_json(scores, join(results_path, 'all_scores_results.json'), mode='w')

    # Scores as DataFrame
    pd_scores: pd.DataFrame = results_as_pandas(scores)

    pd_scores.to_csv(join(results_path, 'all_scores_results.csv'))

    print(pd_scores)
