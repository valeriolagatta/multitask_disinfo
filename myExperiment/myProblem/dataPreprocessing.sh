#!/usr/bin/env bash

###########################################
# Data preprocessing pipeline for MT-DNN. #
# By lui9i								  #
###########################################

## dump original data into tsv
python myExperiment/myProblem/myProblem_prepro.py

## 'bert-base-uncased' 'distilbert-base-uncased' 'roberta-base' 'microsoft/deberta-base' 't5-base'
declare -a PLMS=('bert-base-uncased' 'distilbert-base-uncased' 'roberta-base')

# prepro MyDataset data for all PLMs.
for plm in "${PLMS[@]}"
do
  echo "Prepro MyDataset for $plm"
  python prepro_std.py --model $plm --root_dir data/canonical_data --task_def /content/drive/MyDrive/MY-MT-DNN_Okay/myExperiment/myProblem/myProblem_task_def.yml --workers 32
done
