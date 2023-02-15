#!/bin/bash
for id in 0
do
        python experiments/experiments_performing/one_experiment.py --dataset_name 14lap --id $id --save_dir_name bert_last_element_endpoint_crf -agg rnn
        find . -name 'model.pth' -exec rm -rf {} \;
done
for id in 0
do
        python experiments/experiments_performing/one_experiment.py --dataset_name 14res --id $id  --save_dir_name bert_last_element_endpoint_crf -agg rnn
        find . -name 'model.pth' -exec rm -rf {} \;
done
for id in 0
do
        python experiments/experiments_performing/one_experiment.py --dataset_name 15res --id $id --save_dir_name bert_last_element_endpoint_crf -agg rnn
        find . -name 'model.pth' -exec rm -rf {} \;
done
for id in 0
do
        python experiments/experiments_performing/one_experiment.py --dataset_name 16res --id $id  --save_dir_name bert_last_element_endpoint_crf -agg rnn
        find . -name 'model.pth' -exec rm -rf {} \;
done