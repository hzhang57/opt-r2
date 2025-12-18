CUDA_VISIBLE_DEVICES=1,2,3,4,5,7 accelerate launch \
	--config_file=examples/accelerate_configs/deepspeed_zero3.yaml \
	examples/scripts/grpo_vlm.py \
	--model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
	--output_dir grpo-Qwen3-VL-8B-Instruct \
	--learning_rate 1e-5 \
	--gradient_checkpointing \
	--dtype bfloat16 \
	--max_completion_length 256 \
        --use_vllm \
        --vllm_mode colocate \
        --use_peft \
        --lora_target_modules "q_proj", "v_proj" \
       	--log_completions
