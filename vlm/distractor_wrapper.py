import json
import os
import sys
from typing import Optional

# Support both package import (vlm.distractor_wrapper) and direct execution.
if __package__ is None:
    sys.path.append(os.path.dirname(__file__))

try:
    from .distractor_prompt import DISTRACTOR_PROMPT_3  # package import
except ImportError:
    from distractor_prompt import DISTRACTOR_PROMPT_3  # script import

try:
    # When imported as part of the package (e.g. from vlm.distractor_wrapper)
    from .qwen3_hf import Qwen3HF
except ImportError:
    # Fallback for running the file directly (python vlm/distractor_wrapper.py)
    from qwen3_hf import Qwen3HF

class distractor(object):
    def __init__(self, vlm_model, distractor_prompt):
        self.vlm_model = vlm_model
        self.prompt = distractor_prompt

    def generate(self, question, answer, video: Optional[str] = None, image: Optional[str] = None):
        user_prompt = json.dumps({"Question": question, "Answer": answer}, ensure_ascii=False)
        all_prompt = self.prompt + "\n" + user_prompt
        #print(f"DEBUG: {all_prompt}")
        if video is not None:
            return self.vlm_model.generate(all_prompt, video=video)
        elif image is not None:
            return self.vlm_model.generate(all_prompt, image=image)
        else:
            raise ValueError("Either video or image must be provided")


# test
if __name__ == "__main__":
    default_video = "https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4"
    default_image = "https://ultralytics.com/images/bus.jpg"
    vlm_model = Qwen3HF()
    distractor_prompt = DISTRACTOR_PROMPT_3
    distractor = distractor(vlm_model, distractor_prompt)
    video_question = "who is the primary person in the video?"
    video_answer = "baby"
    print(distractor.generate(question=video_question, answer=video_answer, video=default_video))
    image_question = "where does this happend?"
    image_answer = "spain"
    print(distractor.generate(question=image_question, answer=image_answer, image=default_image))
