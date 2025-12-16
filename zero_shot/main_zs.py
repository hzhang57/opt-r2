import json
import os
import sys
import time
import argparse
from tqdm import tqdm
from PIL import Image

# Add the project root (parent of current directory) to sys.path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_root)

from utils.mm_star import load_open_compass
from utils.mm_star import format_mmstar_dataset_oc
from utils.data_utils import convert_model_name

#from vlm.qwen3_hf import Qwen3HF
from vlm.qwen3_swift import Qwen3VL

from vlm.distractor_prompt import DISTRACTOR_PROMPT_3
from vlm.distractor_wrapper import distractor
from utils.data_utils import create_options_dict

def main(args):
    # 1. Load the dataset, 加载数据
    dataset = load_open_compass(args.oc_dataset)
    n_vals  = len(dataset)
    print("[INFO: Dataset] Loaded {}, with {} samples".format(args.oc_dataset, n_vals))
    # 2. Format the dataset, 数据格式转换
    data_repo = "../../data_repo/"
    data_file = data_repo + "K0_mmstar.json"
    image_folder = data_repo + "K0_mmstar_images/"
    if "MMStar" in args.oc_dataset:
        m0_json = format_mmstar_dataset_oc(dataset, data_file, image_folder).format()
        # save json inot beautiful json format
        with open(data_file, 'w') as f:
            json.dump(m0_json, f, indent=4)
            print("[INFO: Dataset] Saved {} to {}".format(args.oc_dataset, data_file))

    # 2. Load the VLM Agent
    #vlm_model = Qwen3HF(model_id=args.vlm)
    vlm_model = Qwen3VL(model_id=args.vlm)
    distractor_prompt = DISTRACTOR_PROMPT_3
    distractor_generator = distractor(vlm_model, distractor_prompt)

    # 3. Generate the hard negative samples
    m1 = convert_model_name(args.vlm)
    m1_file = data_repo + "K1_mmstar_hard_options_{}.json".format(m1)
    print("[INFO： Hard Negative Options] {}".format(m1_file))

    # Load f0 json file
    with open(data_file, 'r') as f:
        m0_json = json.load(f)
        print("[INFO: Dataset] Loaded {} with {} samples".format(data_file, len(m0_json)))
    m1_json = []
    for ii, a_sample in enumerate(tqdm(m0_json, desc="Distractor Generator..")):
        #if ii > 10:
        #    break

        image_name = a_sample["image"]
        image_path = image_folder + image_name
        #image = Image.open(image_path)
        question = a_sample["question"]
        answer_str = a_sample["answer_str"]
        answer = a_sample["answer"]
        #print("[INFO: Distractor Generator] Processing {} {} {}".format(image_name, question, answer_str))
        # Generate distractors using the VLM
        try:
            hard_options = distractor_generator.generate(question=question, answer=answer_str, image=image_path)
            #print("[INFO: Distractor] Generated: {}".format(hard_options))
        
            # Add distractors to the sample
            # convert json string to json
            hard_options = json.loads(hard_options)['negative_options']
            #a_sample["negative_options"] = hard_options['negative_options']
            new_opt2ans = create_options_dict(answer, answer_str, hard_options)
            a_sample['new_opt2ans'] = new_opt2ans

        except Exception as e:
            print("[ERROR: Distractor Generator] {}".format(e))
            new_opt2ans = a_sample['opt2ans']
            continue
        m1_json.append(a_sample)

    # 4. Save the hard negative samples into json file
    with open(m1_file, 'w') as f:
        json.dump(m1_json, f, indent=4)
        print("[INFO: Dataset] Saved {} with {} samples".format(m1_file, len(m1_json)))

    print("Processed Finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    #parser.add_argument("--vlm", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    #parser.add_argument("--vlm", type=str, default="Qwen/Qwen3-VL-32B-Instruct")
    #parser.add_argument("--vlm", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    #parser.add_argument("--vlm", type=str, default="Qwen/Qwen3-VL-8B-Thinking")
    parser.add_argument("--oc-dataset", type=str, default="../../data_repo/MMStar.tsv")
    parser.add_argument("--output_path", type=str, default="../dataset/new_hard_options.json")
    args = parser.parse_args()
    main(args)
