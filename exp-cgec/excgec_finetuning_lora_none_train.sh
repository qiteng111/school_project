#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export CUDA_VISIBLE_DEVICES=5

cd ../LLaMA-Factory/

MODEL_PATH="/mnt/common/intern/qt/school_project/LLaMA-Factory/data/LLMs/Qwen/Qwen2-7B-Instruct"
TRAIN_DATASET="qt_train_exp_cgec"
VALID_DATASET="qt_valid_exp_cgec"
TEMPLATE="qwen"
OUTPUT_DIR="./model/${TEMPLATE}-llm-7b-chat_qt_test"
EXPORT_DIR="../LLM/${TEMPLATE}-llm-7b-chat_qt_test"
input_file="./data/splits/test_out.json"
output_file="./output/output.json"
LOG_FILE="./log/log_none_train.txt"
filepath_hyp="./output/json/output.json"
filepath_ref="./data/splits/test_out_check_fin.json"


# ######### Training #########
# echo "######### Training #########" >> $LOG_FILE
# CUDA_VISIBLE_DEVICES=4 python src/train_bash.py \
#     --stage sft \
#     --do_train True \
#     --model_name_or_path ${MODEL_PATH} \
#     --dataset ${TRAIN_DATASET},${VALID_DATASET} \
#     --template ${TEMPLATE} \
#     --lora_target q_proj,v_proj \
#     --output_dir ${OUTPUT_DIR} \
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
#     --new_special_tokens "<TGT>" \
#     --resize_vocab True \
#     --lora_rank 8 \
#     >> $LOG_FILE 2>&1

# ######### Export Model #########
# echo "######### Exporting Model #########" >> $LOG_FILE
# CUDA_VISIBLE_DEVICES=6 python src/export_model.py \
#     --model_name_or_path ${MODEL_PATH} \
#     --adapter_name_or_path ${OUTPUT_DIR}  \
#     --template ${TEMPLATE} \
#     --finetuning_type lora \
#     --export_dir  ${EXPORT_DIR} \
#     --export_size 2 \
#     --new_special_tokens "<TGT>" \
#     --export_legacy_format false
#     >> $LOG_FILE 2>&1 \

    
cd ../exp-cgec
EXPORT_DIR="../LLaMA-Factory/data/LLMs/deepseek/deepseek-llm-7b-chat"
input_file="./data/splits/test_out.json"
output_file="./output/output_deepseek_none_train.json"
######### Prediction #########
LOG_FILE="../LLaMA-Factory/log/log_none_train.txt"
echo "######### Running Prediction #########" >> $LOG_FILE
CUDA_VISIBLE_DEVICES=7 python predict_none_train.py \
    --input_file ${input_file} \
    --output_file ${output_file} \
    --model_dir ${EXPORT_DIR} \
    >> $LOG_FILE 2>&1


# cd ../exp-cgec
# output_file="../exp-cgec/output/output.json"
# filepath_hyp="../exp-cgec/output/json/output.json"
# filepath_ref="../exp-cgec/data/splits/test_out_check_fin.json"

# ######### Data-process #########
# echo "######### Data-process #########" >> $LOG_FILE
# CUDA_VISIBLE_DEVICES=3 python ./util/data/data-process.py \
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

