CUDA_VISIBLE_DEVICES=7 \
    swift rollout \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --vllm_data_parallel_size 1 \
    --vllm_max_model_len 21000 \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_max_num_seqs 16

#    --model Qwen/Qwen3-VL-32B-Instruct \
#    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
#        --model Qwen/Qwen3-VL-8B-Instruct \
#        --model Qwen/Qwen3-VL-4B-Instruct \
#        --model Qwen/Qwen2.5-VL-3B-Instruct \
