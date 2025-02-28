TIME_TAG=$(date "+%Y%m%d_%H%M%S")

MODEL=ckpts-deepseek
DATA=gsm8k

SHELL_PATH=$(dirname $(readlink -f "$0"))
BASE_PATH=/apdcephfs_cq10/share_1150325/ztjiaweixu/huggingface
OUTPUT_PATH=output/GRPO/${DATA}/${MODEL}/${TIME_TAG}

if [ ! -d "$OUTPUT_PATH" ]; then
    mkdir -p "$OUTPUT_PATH"
fi

export WANDB_MODE=offline
export WANDB_PROJECT=grpo-try
export WANDB_RUN_GROUP=GRPO
export WANDB_NAME=test
export WANDB_DIR=$OUTPUT_PATH
export WANDB_CACHE_DIR=$OUTPUT_PATH/wandb/cache
export WANDB_ARTIFACT_DIR=$OUTPUT_PATH/wandb/artifact

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
MASTER_PORT=29503 \
swift rlhf \
    --rlhf_type grpo \
    --model_type deepseek_r1 \
    --model /sskj-prod/gzs/xjw/models/${MODEL} \
    --external_plugins ${SHELL_PATH}/plugin/plugin.py \
    --reward_funcs external_gsm8k length \
    --use_vllm false \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_max_model_len 8192 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset /sskj-prod/gzs/xjw/datas/${DATA} \
    --custom_register_path ${SHELL_PATH}/dataset/dataset.py \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir $OUTPUT_PATH \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 0.9 \
    --deepspeed zero2 \
    --log_completions true \
    --report_to wandb

# ps -ef | grep swift | awk '{print $2}'| xargs kill -9
