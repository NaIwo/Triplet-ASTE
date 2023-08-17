from aste.dataset.statistics import ResultInvestigator
from aste.dataset.reader import ASTEDataset

from os import getcwd, path, listdir
from os.path import join, basename, splitext

if __name__ == '__main__':
    # You can perform investigation about your results by using ResultInvestigator class.
    for dataset in ['14lap']:
        main_path: str = path.join('experiments', 'experiments_results_newest', dataset, 'transformer', 'amazon')
        label_path: str = join('dataset', 'data', 'ASTE_data_v2', dataset, 'test.txt')
        for variation in listdir(main_path):
            if 'pred' not in variation:
                continue

            pred_path: str = join(main_path, variation)

            base: str = basename(pred_path)
            file_name, extension = splitext(base)
            save_path: str = join(main_path, f'{file_name}_stats')

            # One thing is required to do. Build ASTEDataset from original data and predicted one
            original_data = ASTEDataset(label_path)
            predicted_data = ASTEDataset(pred_path)

            # Pass them to class constructor
            # NOTE: You can pass your own implementation functions to calculate more stats:
            #   ResultInvestigator(
            #                  model_prediction: ASTEDataset,
            #                  result_stats_func: Optional[Dict] = None,
            #                  advanced_result_stats_func: Optional[Dict] = None,
            #                  statistics_func: Optional[Dict] = None,
            #                  phrases_func: Optional[Dict] = None,
            #                  phrases_to_count: Optional[List] = None)
            investigator = ResultInvestigator(original_data=original_data, model_prediction=predicted_data)
            # Now you can compute the result and save to json or csv as well.
            investigator.compute()
            investigator.pprint()
            # print(investigator.to_pandas())
            investigator.to_csv(save_path)
