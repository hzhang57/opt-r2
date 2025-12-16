
#    --model Qwen/Qwen3-VL-4B-Instruct \
#    --model Qwen/Qwen3-VL-8B-Instruct \
#    --model Qwen/Qwen2.5-VL-3B-Instruct \
#MAX_PIXELS=262144 \
WANDB_API_KEY=e15fbbc857d13f6f81dc158724b3bbf8f7dbce2e \
CUDA_VISIBLE_DEVICES=1,2,6 \
NPROC_PER_NODE=3 \
MAX_PIXELS=262144 \
MASTER_PORT=29600 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_r1v_acc format \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8001 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset 'lmms-lab/multimodal-open-r1-8k-verified' \
    --load_from_cache_file true \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 400 \
    --save_steps 400 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --output_dir output/GRPO_GEOQA \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --repetition_penalty 1.1 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero3 \
    --log_completions true \
    --report_to wandb \
    --run_name Qwen3-VL-8B-Instruct-Open-R1 \
    --num_iterations 2 \
    --async_generate false \
    --beta 0.001 \
    --max_grad_norm 0.5 \
