#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
CKPT=result_zs
ORI_OR_NEW=1 # 0 for ori, 1 for gda
DATA_SET="MMStar"
IMAGE_FOLDER="../data_repo/K0_mmstar_images/"
QUESTION_FILES=("../data_repo/K1_mmstar_hard_options_Qwen3_VL_4B_Instruct.json")
#QUESTION_FILES=("../data_repo/K1_mmstar_hard_options_Qwen3_VL_30B_A3B_Instruct.json")
#QUESTION_FILES=("../data_repo/K1_mmstar_hard_options_Qwen3_VL_32B_Instruct.json")
#QUESTION_FILES=("../data_repo/K1_mmstar_hard_options_Qwen3_VL_8B_Instruct.json")

# Define the models to loop over
#MODEL_PATHES=("Qwen/Qwen2.5-VL-7B-Instruct")
#MODEL_PATHES=("Qwen/Qwen2.5-VL-3B-Instruct")
#MODEL_PATHES=("OpenBMB/MiniCPM-V-2_6")
#MODEL_PATHES=("OpenGVLab/InternVL3_5-4B")

MODEL_PATHES=("deepseek-ai/Janus-1.3B")

#MODEL_PATHES=("lmms-lab/LLaVA-OneVision-1.5-4B-Instruct")
#MODEL_PATHES=("OpenBMB/MiniCPM-V-4_5")
#MODEL_PATHES=("deepseek-ai/DeepSeek-OCR")


##MODEL_PATHES=("ZhipuAI/GLM-4.1V-9B-Base")
##MODEL_PATHES=("google/gemma-3n-E4B")
##MODEL_PATHES=("deepseek-ai/deepseek-vl2-small")

# 遍历数组时需使用正确语法，并引用数组中的每个元素
for MODEL_PATH in "${MODEL_PATHES[@]}"; do
    for QUESTION_FILE in "${QUESTION_FILES[@]}"; do
        # 从 MODEL_PATH 中提取模型名称（取最后一个 '/' 之后的部分）
        model_name=$(basename "$MODEL_PATH")
        # 从 QUESTION_FILE 中提取文件名称（去掉 .json 后缀）
        question_name=$(basename "$QUESTION_FILE" .json)
        EXP_NAME="${model_name}_ORI_OR_NEW_${ORI_OR_NEW}_x${NUM_FRAMES}_${question_name}"
        # Run for the Query JSON file
        python evaluate_mmstar.py \
            --model-path "$MODEL_PATH" \
            --max_new_tokens 20 \
            --question-file "$QUESTION_FILE" \
            --image-folder "$IMAGE_FOLDER" \
            --answers-file "./outeval_mmstar/${EXP_NAME}.json" \
            --temperature 0 \
	        --ori-or-new ${ORI_OR_NEW} \
            --conv-mode "$CONV_MODE" > "./outeval_mmstar/${EXP_NAME}.log"
    done
done
