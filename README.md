# README3

Placeholder for additional setup or usage notes. Add project-specific instructions here as needed.

## Install dependencies

You can install all required dependencies using the requirements.txt file:

```bash
pip install -r requirements.txt
```

Or install them individually:

```bash
pip install "ms-swift==3.10.2"
pip install "transformers==4.57.1" 
pip install "qwen_vl_utils>=0.0.14"
pip install "trl[vllm]"
pip install math_verify
pip install wandb
pip install weave
pip install packaging ninja
pip install deepspeed
pip install "vllm==0.11.0"
pip install msgspec
pip install torchvision
pip install decord
pip install pandas
pip install tqdm
pip install Pillow
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

If evaluate deepseek-ocr model, need install "transformers==4.46.3, easydict"