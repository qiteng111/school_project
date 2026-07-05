#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export CUDA_VISIBLE_DEVICES=1

cd ../LLaMA-Factory/

MODEL_PATH="/home/s202507015/workspace/models/modelscope/models/qwen/Qwen1___5-7B-Chat"
TRAIN_DATASET_1="qt_train_exp_cgec_step1"
VALID_DATASET_1="qt_valid_exp_cgec_step1"

TRAIN_DATASET_2="qt_train_exp_cgec_step2"
VALID_DATASET_2="qt_valid_exp_cgec_step2"

TEMPLATE="qwen"
OUTPUT_DIR_1="./model/${TEMPLATE}-llm-7b-chat_qt_1.5_step1"
EXPORT_DIR_1="../LLM/${TEMPLATE}-llm-7b-chat_qt_1.5_step1"

OUTPUT_DIR_2="./model/${TEMPLATE}-llm-7b-chat_qt_1.5_step2"
EXPORT_DIR_2="../LLM/${TEMPLATE}-llm-7b-chat_qt_1.5_step2"

input_file_1="./data/splits/test_out_qt.json"

output_file_1="./output/step1_output/output_qwen_1.5_step1.json"
output_file_2="./output/step2_output/output_qwen_1.5_step2.json"

output_file_1_processed="./output/step1_output/output_qwen_1.5_step1_processed.json"
output_file_2_processed="./output/step2_output/output_qwen_1.5_step2_processed.json"

LOG_FILE="./log/log1.5_step2.txt"

filepath_hyp_1="./output/json/output_qwen_1.5_step1.json"
filepath_hyp_2="./output/json/output_qwen_1.5_step2.json"

filepath_ref_1="./data/splits/test_out_check_fin_qt_1.json"
filepath_ref_2="./data/splits/test_out_check_fin_qt_2.json"


# ######### Training-step1 #########
# echo "######### Training #########" >> $LOG_FILE
# CUDA_VISIBLE_DEVICES=1 python src/train_bash.py \
#     --stage sft \
#     --do_train True \
#     --model_name_or_path ${MODEL_PATH} \
#     --dataset ${TRAIN_DATASET_1},${VALID_DATASET_1} \
#     --template ${TEMPLATE} \
#     --lora_target q_proj,v_proj \
#     --output_dir ${OUTPUT_DIR_1} \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --cutoff_len 1024 \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --lr_scheduler_type cosine \
#     --logging_steps 10 \
#     --warmup_steps 20 \
#     --save_steps 100 \
#     --eval_steps 100 \
#     --evaluation_strategy steps \
#     --load_best_model_at_end \
#     --learning_rate 5e-5 \
#     --num_train_epochs 3.0 \
#     --finetuning_type lora \
#     --plot_loss \
#     --val_size 0.1116 \
#     --fp16 \
#     --lora_rank 8 \
#     >> $LOG_FILE 2>&1

# ######### Export Model #########
# echo "######### Exporting Model #########" >> $LOG_FILE
# CUDA_VISIBLE_DEVICES=1 python src/export_model.py \
#     --model_name_or_path ${MODEL_PATH} \
#     --adapter_name_or_path ${OUTPUT_DIR_1}  \
#     --template ${TEMPLATE} \
#     --finetuning_type lora \
#     --export_dir  ${EXPORT_DIR_1} \
#     --export_size 2 \
#     --export_legacy_format false
#     >> $LOG_FILE 2>&1 \


cd ../exp-cgec
######### Prediction-step1 #########
echo "######### Running Prediction #########" >> $LOG_FILE
CUDA_VISIBLE_DEVICES=1 python predict_step1.py \
    --input_file ${input_file_1} \
    --output_file ${output_file_1} \
    --model_dir ${EXPORT_DIR_1} \
    >> $LOG_FILE 2>&1


# ######### Data-process-step1 #########
# cd ../exp-cgec
# echo "######### Data-process #########" >> $LOG_FILE
# CUDA_VISIBLE_DEVICES=3 python ./util/data/data-process-step1.py \
#     --input_file ${output_file_1} \
#     --output_file ${output_file_1_processed} \
#     >> $LOG_FILE 2>&1

# ######### Training-step2 #########
# echo "######### Training #########" >> $LOG_FILE
# CUDA_VISIBLE_DEVICES=1 python src/train_bash.py \
#     --stage sft \
#     --do_train True \
#     --model_name_or_path ${MODEL_PATH} \
#     --dataset ${TRAIN_DATASET_2},${VALID_DATASET_2} \
#     --template ${TEMPLATE} \
#     --lora_target q_proj,v_proj \
#     --output_dir ${OUTPUT_DIR_2} \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --cutoff_len 1024 \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --lr_scheduler_type cosine \
#     --logging_steps 10 \
#     --warmup_steps 20 \
#     --save_steps 100 \
#     --eval_steps 100 \
#     --evaluation_strategy steps \
#     --load_best_model_at_end \
#     --learning_rate 5e-5 \
#     --num_train_epochs 3.0 \
#     --finetuning_type lora \
#     --plot_loss \
#     --val_size 0.1116 \
#     --fp16 \
#     --resize_vocab True \
#     --lora_rank 8 \
#     >> $LOG_FILE 2>&1

# ######### Export Model #########
# echo "######### Exporting Model #########" >> $LOG_FILE
# CUDA_VISIBLE_DEVICES=1 python src/export_model.py \
#     --model_name_or_path ${MODEL_PATH} \
#     --adapter_name_or_path ${OUTPUT_DIR_2}  \
#     --template ${TEMPLATE} \
#     --finetuning_type lora \
#     --export_dir  ${EXPORT_DIR_2} \
#     --export_size 2 \
#     --export_legacy_format false
#     >> $LOG_FILE 2>&1 \


# ######### Prediction-step2 #########
# echo "######### Running Prediction #########" >> $LOG_FILE
# CUDA_VISIBLE_DEVICES=1 python predict_step2.py \
#     --input_file ${output_file_1_processed} \
#     --output_file ${output_file_2} \
#     --model_dir ${EXPORT_DIR_2} \
#     >> $LOG_FILE 2>&1


# ######### Data-process-step2 #########
# cd ../exp-cgec
# echo "######### Data-process #########" >> $LOG_FILE
# CUDA_VISIBLE_DEVICES=3 python ./util/data/data-process-step2.py \
#     --input_file ${output_file_2} \
#     --output_file ${output_file_2_processed} \
#     >> $LOG_FILE 2>&1


# cd ../exp-cgec
# output_file="./output/output_qwen_1.5.json"
# filepath_hyp="./output/json/output_qwen_1.5.json"
# filepath_ref="./data/splits/test_out_check_fin_qt.json"


# ######### Data-process #########
# echo "######### Data-process #########" >> $LOG_FILE
# CUDA_VISIBLE_DEVICES=3 python ./util/data/data-process_qt.py \
#     --input_file ${output_file} \
#     --output_file ${filepath_hyp} \
#     >> $LOG_FILE 2>&1


# ######### Evaluation #########
# echo "######### Running Evaluation #########" >> $LOG_FILE
# CONDA_BASE=$(conda info --base)
# source "$CONDA_BASE/etc/profile.d/conda.sh"
# conda activate excgec-eval
# python evaluation.py \
#     --filepath_hyp ${filepath_hyp} \
#     --filepath_ref ${filepath_ref} \
#     >> $LOG_FILE 2>&1 

# conda deactivate

