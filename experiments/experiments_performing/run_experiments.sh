#!/bin/bash
for id in 1
do
        python experiments/experiments_performing/one_experiment.py --dataset_name 14lap --id $id --save_dir_name crf_sigmoid10 -agg endpoint
done
for id in 1
do
        python experiments/experiments_performing/one_experiment.py --dataset_name 14res --id $id  --save_dir_name crf_sigmoid10 -agg endpoint
done
for id in 1
do
        python experiments/experiments_performing/one_experiment.py --dataset_name 15res --id $id --save_dir_name crf_sigmoid10 -agg endpoint
done
for id in 1
do
        python experiments/experiments_performing/one_experiment.py --dataset_name 16res --id $id  --save_dir_name crf_sigmoid10 -agg endpoint
done