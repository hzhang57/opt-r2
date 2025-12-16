import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from PIL import Image
import math
from torch.utils.data import Dataset, DataLoader

import requests
from io import BytesIO
import re
from pathlib import Path

import os
from typing import List, Literal
import random
import numpy as np

from utils.data_utils import TypeAccuracy_ABCD, pretty_print_dict, generate_conversations
from vlm.qwen3_swift import Qwen3VL

random.seed(42)
def parse_first_number(text):
    """
    This function takes a string as input and returns the first number it finds in the string.
    If no number is found, it returns None.
    
    :param text: The input string to search for a number
    :return: The first number found as a string, or None if no number is found
    """
    # Use regular expression to find the first number in the string
    match = re.search(r'\d+', text)
    
    # Return the matched number if found, otherwise return None
    return match.group() if match else None


def convert_parenthesized_digits(s):
    """
    Convert parenthesized digits (0)-(3) to letters (A)-(D).
    
    Args:
        s (str): Input string containing parenthesized digits to convert
        
    Returns:
        str: String with (0)-(3) converted to (A)-(D) respectively
    """
    # Create mapping dictionary
    #mapping = {'(0)': '(A)', '(1)': '(B)', '(2)': '(C)', '(3)': '(D)'}
    mapping = {'(0)': 'A.', '(1)': 'B.', '(2)': 'C.', '(3)': 'D.'}
    
    # Replace each occurrence
    result = s
    for old, new in mapping.items():
        result = result.replace(old, new)
    
    return result


def main(args):
    # Load Model
    #disable_torch_init()
    ### Load Model
    vlm = Qwen3VL(model_id=args.model_path)
    # Load Questions
    annotations = json.load(open(os.path.expanduser(args.question_file), "r"))

    # Overall Accuracy for All Questions
    correct = 0
    total = 0

    global_acc = TypeAccuracy_ABCD("Global")
    qa1_acc = TypeAccuracy_ABCD("coarse perception")
    qa2_acc = TypeAccuracy_ABCD("fine-grained perception")
    qa3_acc = TypeAccuracy_ABCD("instance reasoning")
    qa4_acc = TypeAccuracy_ABCD("logical reasoning")
    qa5_acc = TypeAccuracy_ABCD("science & technology")
    qa6_acc = TypeAccuracy_ABCD("math")

    ii = 0
    out_json = []
    for line in tqdm(annotations, total=len(annotations)):
        #if ii > 50:
        #    break
        #ii+=1
        # Q-A Pair
        idx = line["id"]
        quest_type = line["quest_type"]

        if args.ori_or_new == 0: # original
            conversations = generate_conversations(line["question"], line["opt2ans"], line["answer"])
            qs = conversations[0]["value"]
            gt_answer   = conversations[1]["value"]

        elif args.ori_or_new == 1: # gda
            conversations = generate_conversations(line["question"], line["new_opt2ans"], line["answer"])
            qs = conversations[0]["value"]
            gt_answer   = conversations[1]["value"]

        with torch.inference_mode():
            image_path = os.path.join(args.image_folder, line["image"])
            outputs = vlm.generate(text=qs, image=image_path)        
            #outputs = vlm.generate_text(text=qs)        
            print("[INFO: QA] {} {}\n{}".format(idx, qs, outputs))

        # Decode output
        outputs = outputs.strip()
        total += 1
        answer_idx = outputs
        global_acc.update(gt_answer, answer_idx)

        ## Add result
        if args.ori_or_new == 0: # original
            line["AI"] = outputs
            line["Corret_OR_Wrong"] = [1 if "{}.".format(outputs) == gt_answer[:2] else 0]
        elif args.ori_or_new == 1: # gda
            line["NEW-AI"] = outputs
            line["NEW Corret_OR_Wrong"] = [1 if "{}.".format(outputs) == gt_answer[:2] else 0]

        out_json.append(line)
        if "coarse perception" == quest_type:
            qa1_acc.update(gt_answer, answer_idx)
        elif "fine-grained perception" == quest_type:
            qa2_acc.update(gt_answer, answer_idx)
        elif "instance reasoning" == quest_type:
            qa3_acc.update(gt_answer, answer_idx)
        elif "logical reasoning" == quest_type:
            qa4_acc.update(gt_answer, answer_idx)
        elif "science & technology" == quest_type:
            qa5_acc.update(gt_answer, answer_idx)
        elif "math" == quest_type:
            qa6_acc.update(gt_answer, answer_idx)
        else:
            print(f"Unknown Type: {idx}")
        # print each type accuracy
        print("-----"*5)
        qa1_acc.print_accuracy()
        qa2_acc.print_accuracy()
        qa3_acc.print_accuracy()
        qa4_acc.print_accuracy()
        qa5_acc.print_accuracy()
        qa6_acc.print_accuracy()
        #global_acc.print_accuracy()
        print("-----"*5)
        # average over type
        avg_acc = (qa1_acc.get_accuracy() + qa2_acc.get_accuracy() + qa3_acc.get_accuracy() + qa4_acc.get_accuracy() + qa5_acc.get_accuracy() + qa6_acc.get_accuracy()) / 6.0
        print("Average Acc over Type: {:.4f}".format(avg_acc))

    with open(args.answers_file, "w") as f:
        json.dump(out_json, f, indent=2)
    print("Process Finished")

def parse_answer(outputs):
    if "Answer is:" in outputs:
    # with graph
        outputs = outputs.split("Answer is: ")[-1]
    if "answer is " in outputs:
    # with graph
        outputs = outputs.split("answer is ")[-1].strip(".")
    # remove graph
    answer_id = outputs[0]
    try:
        answer_id = int(answer_id)
        return answer_id
    except:
        return -1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.json")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--ori-or-new", type=int, default=0) # 0 original, 1 gda
    args = parser.parse_args()
    main(args)
