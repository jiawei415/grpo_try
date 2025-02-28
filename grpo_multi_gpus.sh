BASE_PATH=/apdcephfs_cq10/share_1150325/ztjiaweixu/huggingface
MODEL=Llama-3.2-3B-Instruct
DATA=NuminaMath-TIR

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
MASTER_PORT=29500 \
swift rlhf \
    --rlhf_type grpo \
    --model ${BASE_PATH}/${MODEL} \
    --reward_funcs accuracy format cosine repetition \
    --use_vllm false \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_max_model_len 8192 \
    --train_type full \
    --torch_dtype float16 \
    --dataset ${DATA_PATH}/${DATA}#5000 \
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
    --output_dir output/GRPO/${DATA}_${MODEL} \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 0.9 \
    --system prompt.txt \
    --deepspeed zero2 \
    --log_completions true

# ps -ef | grep swift | awk '{print $2}'| xargs kill -9
