#!/usr/bin/env bash

############################################## 
# Training script for myProblem.			 #
# It supports single and multi-task training #
# By Luigi									 #
############################################## 

set -e


if [[ $# -lt 6 ]]; then
  echo "It requires 6 args to run the script and the current # of bash args: $#"
  echo "run_myproblem_finetune.sh <data_dir> <model_type> <model_size> <task> <batch_size> <num_gpus> <epochs> <learning_rate>"
  exit 1
fi

data_dir=$1
echo "Data dir: ${data_dir}"
model_type=$2
echo "Model type: ${model_type}"
model_size=$3
echo "Model size: ${model_size}"
# training set
task=$4
echo "Task: ${task}"
batch_size=$5
echo "Batch-size: ${batch_size}"
num_gpus=$6
echo "GPU counts: ${num_gpus}"
EPOCH=$7
echo "epochs: ${EPOCH}"
LR=$8
echo "learning rate: ${LR}"

export ROOT_DIR="/content/drive/MyDrive/MY-MT-DNN_Okay"
export OPTIM="adamax"
export TASK_DEF="myExperiment/myProblem/myProblem_task_def.yml"
export BS=${batch_size}

echo "myProblem_task_def.yml path: ${TASK_DEF}"

# train task
if [ ${task} == "mtdnn" ]; then
train_dataset="fnd,sqdc"
test_dataset="fnd,sqdc"
else
echo "Specify the folders that contain the datasets of the respective tasks!"
fi


echo "Training data: ${train_dataset}_train.json"
echo "Dev data: ${test_dataset}_dev.json"


if [ ${model_type} == "distilbert" ]; then
  MD="distilbert-${model_size}-uncased"
  DD="distilbert-${model_size}-uncased"
  ED=1
elif [ ${model_type} == "bert" ]; then
  MD="bert-${model_size}-uncased"
  DD="bert-${model_size}-uncased"
  ED=1
elif [ ${model_type} == "roberta" ]; then
  MD="roberta-${model_size}"
  DD="roberta-${model_size}"
  ED=2
elif [ ${model_type} == "deberta" ]; then
  MD="microsoft/deberta-${model_size}"
  DD="microsoft/deberta-${model_size}"
  ED=6
elif [ ${model_type} == "t5e" ]; then
  MD="t5-${model_size}"
  DD="t5-${model_size}"
  ED=8
elif [ ${model_type} == "electra" ]; then
  MD="google/electra-${model_size}-discriminator"
  DD="bert-base-uncased"
  ED=7
elif [ ${model_type} == "mtdnn" ]; then
  MD="mt_dnn_modes/mt_dnn_${model_size}_uncased.pt"
  DD="bert-${model_size}-uncased"
  ED=1
else
  echo "Unknown model ${model_type}"
  exit 1
fi

output_dir="${ROOT_DIR}/${task}/${DD}"
echo "Output directory: $output_dir"
mkdir -p ${output_dir}

if [[ -f "${output_dir}/model*.pt" ]]; then
 rm "${output_dir}/model*.pt"
 rm "${output_dir}/config.json"
fi

echo "Training ${task} tokenized by ${DD} with ${MD}"

LOG_FILE="${output_dir}/mt-dnn-train.log"
#
if [ ${num_gpus} -ge 2 ]; then
 # multi gpu training
 # DDP config
 export MASTER_ADDR=localhost
 export MASTER_PORT="8787"
 export NNODES=1
 export NODE_RANK=0
 export GPUS_PER_NODE=${num_gpus}
 export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
 export DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
 python -m torch.distributed.launch $DISTRIBUTED_ARGS train.py --data_dir=${data_dir}/${DD} --task_def=${TASK_DEF}  --train_dataset=${train_dataset} --test_dataset=${test_dataset} --init_checkpoint=${MD} --batch_size=${BS} --learning_rate=${LR} --epochs=${EPOCH} --encoder_type=${ED} --optimizer=${OPTIM} --output_dir=${output_dir} --log_file=${LOG_FILE}
else
 python train.py --data_dir=${data_dir}/${DD} --task_def=${TASK_DEF} --train_dataset=${train_dataset} --test_dataset=${test_dataset} --init_checkpoint=${MD} --batch_size=${BS} --learning_rate=${LR} --epochs=${EPOCH} --encoder_type=${ED} --optimizer=${OPTIM} --output_dir=${output_dir} --log_file=${LOG_FILE}
fi
