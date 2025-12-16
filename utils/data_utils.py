import base64
import io

import re
import os
from typing import Tuple, Dict
from PIL import Image
import json


def pretty_print_dict(d):
    print(json.dumps(d, indent=4))


def generate_conversations(question, opt2ans, answer):
    """
    Generates a conversation from a question and options.
    """
    opt_str = ""
    for k in opt2ans.keys():
        opt_str += "{}. {}\n".format(k, opt2ans[k])

    new_question = "Question:\n{}\nOption:\n{}\nReturn only the option letter of the correct answer (e.g. A, B, C, D)".format(question, opt_str)
    gt_answer  = "{}. {}".format(answer, opt2ans[answer])
    conversations = [
        {
            'from': "human",
            'value': new_question
        },
        {
            'from': "gpt",
            'value': gt_answer
        }
    ]
    return conversations


def create_options_dict(answer, answer_str, wrong_options):
    """
    Combine correct and wrong options into a dictionary with letters as keys.
    
    Args:
        answer: The correct answer letter (e.g., 'A', 'B', 'C', 'D')
        answer_str: The correct answer string content
        wrong_options: List of wrong answer strings
    
    Returns:
        dict: Dictionary with letter keys and string values
    
    Example:
        >>> result = create_options_dict('B', 'Paris', ['London', 'Berlin', 'Madrid'])
        >>> result
        {'A': 'London', 'B': 'Paris', 'C': 'Berlin', 'D': 'Madrid'}
    """
    # Generate enough letters for all options
    total_options = len(wrong_options) + 1  # +1 for correct answer
    letters = [chr(65 + i) for i in range(total_options)]  # A, B, C, D, ...
    
    # Create dictionary with correct answer at the specified letter
    options_dict = {answer: answer_str}
    
    # Add wrong options to remaining letters (excluding the correct answer letter)
    available_letters = [l for l in letters if l != answer]
    
    for i, wrong_str in enumerate(wrong_options):
        options_dict[available_letters[i]] = wrong_str
    # Sort dictionary by key
    options_dict = dict(sorted(options_dict.items()))
    return options_dict


def save_image_to_folder(image: Image.Image, folder_path: str, index: int, prefix: str = "img_", ext: str = "jpg", format: str = None):
    """
    Saves a PIL Image to the specified folder.
    
    Args:
        image:       The PIL Image object to save.
        folder_path: The path to the folder where the image will be saved.
        filename:    The name of the file (e.g. "my_image.jpg"). 
                     If None, defaults to "image.<ext>".
        format:      Optional format override (e.g. "JPEG", "PNG"). 
                     If None, inferred from filename extension or image.format.
    """
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Determine filename
    # fallback extension
    ext = (format or image.format or "PNG").lower()
    filename = f"{prefix}{index:05d}.{ext}"

    # Full path
    save_path = os.path.join(folder_path, filename)

    # Save
    image.save(save_path, format=format)
    print(f"Saved image to {save_path}")
    return filename

import re
from typing import Optional, Tuple, Dict, Any

def save_image_to_folder_base64(base64_string:str, folder_path: str, index: int, prefix: str = "img_", ext: str = "jpg", format: str = None):
    """
    Saves a PIL Image to the specified folder.
    
    Args:
        image:       The PIL Image object to save.
        folder_path: The path to the folder where the image will be saved.
        filename:    The name of the file (e.g. "my_image.jpg"). 
                     If None, defaults to "image.<ext>".
        format:      Optional format override (e.g. "JPEG", "PNG"). 
                     If None, inferred from filename extension or image.format.
    """
    image_data = base64.b64decode(base64_string)
        
    # Create PIL Image from bytes
    image = Image.open(io.BytesIO(image_data))
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Determine filename
    # fallback extension
    ext = (format or image.format or "PNG").lower()
    filename = f"{prefix}{index:05d}.{ext}"

    # Full path
    save_path = os.path.join(folder_path, filename)

    # Save
    image.save(save_path, format=format)
    print(f"Saved image to {save_path}")
    return filename



def split_question_into_query_and_options_0(text):
    """
    Given a string of the form:
      "<Question text>\nOptions: A: optA, B: optB, C: optC, …"
    returns (question, options_dict) where
      question is the text before "Options:",
      options_dict maps 'A'→"optA", 'B'→"optB", etc.
    """
    # 1. Split into question and options
    try:
        question_part, options_part = text.split('Options:', 1)
    except ValueError:
        raise ValueError("Input must contain 'Options:'")

    question = question_part.strip()

    # 2. Use regex to find all "Key: Value" pairs
    pattern = r'([A-Z]):\s*(.*?)(?=(?:,\s*[A-Z]:)|$)'
    matches = re.findall(pattern, options_part)

    # 3. Build the dict, stripping any trailing commas/spaces
    options = {
        key: val.strip().rstrip(',')
        for key, val in matches
    }

    return question, options

def split_question_into_query_and_options_1(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Extracts the question and options from a given text.

    Parameters:
    -----------
    text : str
        The input string containing a question and options.

    Returns:
    --------
    Tuple[str, Dict[str, str]]
        A tuple containing:
        - The question string.
        - A dictionary mapping option letters to option texts.
    """
    # Extract the question text after 'Question:' up to the next line break
    question_match = re.search(r'Question:\s*(.*?)\s*(?=\r?\n)', text, re.IGNORECASE)
    question = question_match.group(1).strip() if question_match else ''

    # Extract the block of text after 'Choices:' as the options block
    options_block_match = re.search(r'Choices:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    options_block = options_block_match.group(1).strip() if options_block_match else ''

    # Parse each line in the options block for option letter and text
    options: Dict[str, str] = {}
    for line in options_block.splitlines():
        line = line.strip()
        if not line:
            continue
        # Match formats like "(A) text", "A) text", "A. text", etc.
        opt_match = re.match(r'^\(?\s*([A-Za-z])\s*\)?[\.\):]?\s*(.+)$', line)
        if opt_match:
            key = opt_match.group(1)
            value = opt_match.group(2).strip()
            options[key] = value

    return question, options



def split_question_into_query_and_options_2(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Given a text with a question followed by an "Options:" section,
    returns a tuple (question, options_dict), where:
      - question is the question string (without the "Options:" part)
      - options_dict maps each option letter (e.g. "A") to its code snippet.
    """
    # 1. Split off the question
    parts = text.split("\nOptions:", 1)
    question = parts[0].strip()
    options_block = parts[1] if len(parts) > 1 else ""

    # 2. Normalize separators between options
    #    In the example, options are sometimes separated by ", B:" etc.
    #    We insert a newline before each "<Letter>:"
    options_block = re.sub(r',\s*([A-Za-z]:)', r'\n\1', options_block)

    # 3. Extract each option letter and its content
    pattern = re.compile(r'([A-Za-z]):\s*(.*?)(?=(?:\n[A-Za-z]:)|\Z)', re.DOTALL)
    options: Dict[str, str] = {
        m.group(1): m.group(2).strip()
        for m in pattern.finditer(options_block)
    }

    return question, options


def split_question_into_query_and_options_3(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Extracts the question and options from a text block of the form:
      [optional hint lines]
      Question: <question text>
      (A) <option A>
      (B) <option B>
      (C) <option C>
      (D) <option D>
    Returns:
      - question: the question string (without the "Question:" prefix)
      - options: dict mapping "A", "B", "C", "D" to their corresponding texts
    """
    question = ''
    options: Dict[str, str] = {}

    # Process line by line
    for line in text.splitlines():
        line = line.strip()
        # Extract question
        if line.lower().startswith('question:'):
            question = line[len('Question:'):].strip()
            continue
        # Extract options of the form "(X) value"
        m = re.match(r'^\(([A-Za-z])\)\s*(.+)$', line)
        if m:
            key = m.group(1)
            value = m.group(2).strip()
            options[key] = value

    return question, options


def remap_options_to_indices(options):
    """
    Given options like {'A': ..., 'B': ..., …},
    returns {0: optionA, 1: optionB, …}.
    """
    return {ord(letter) - ord('A'): text
            for letter, text in options.items()}


class TypeAccuracy(object):
    def __init__(self, type_name):
        self.correct = 0
        self.total = 10e-9
        self.type_name = type_name

    def update(self, gt, pred):
        self.total += 1
        if "({})".format(pred) in gt:
            self.correct += 1
            print("### COREECT ### : GT {} / Pred ({}) ".format(gt, pred))
        else:
            print("### WRONG ### : GT {} / Pred ({}) ".format(gt, pred))

    def get_accuracy(self):
        return 1.0*self.correct / self.total

    def print_accuracy(self):
        print("{} Accuracy: {:.4f} | {}/{}".format(
                self.type_name,
                self.get_accuracy(),
                self.correct,
                self.total
            ))

class TypeAccuracy_ABCD(object):
    def __init__(self, type_name):
        self.correct = 0
        self.total = 10e-9
        self.type_name = type_name

    def update(self, gt, pred):
        self.total += 1
        if "{}.".format(pred) == gt[:2]:
        #if pred == gt:
            self.correct += 1
            print("### COREECT ### : GT {} / Pred ({}) ".format(gt, pred))
        else:
            print("### WRONG ### : GT {} / Pred ({}) ".format(gt, pred))

    def get_accuracy(self):
        return 1.0*self.correct / self.total

    def print_accuracy(self):
        print("{} Accuracy: {:.4f} | {}/{}".format(
                self.type_name,
                self.get_accuracy(),
                self.correct,
                self.total
            ))


def convert_model_name(model_string):
    """
    Convert a model string from format "provider/model-name" to "model_name".
    
    Args:
        model_string (str): Input string in format like "openai/gpt-oss-120b"
    
    Returns:
        str: Converted string in format like "gpt_oss_120b"
    
    Example:
        >>> convert_model_name("openai/gpt-oss-120b")
        'gpt_oss_120b'
    """
    # Split by '/' and take the part after the slash
    model_part = model_string.split('/')[-1]
    
    # Replace hyphens with underscores
    converted_name = model_part.replace('-', '_')
    
    return converted_name