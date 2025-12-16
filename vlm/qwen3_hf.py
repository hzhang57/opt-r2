"""Minimal Hugging Face Transformers wrapper for Qwen3-VL (image/video + text)."""
import os
import tempfile
from typing import Any, Dict, List, Optional

import requests
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class Qwen3HF:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
        device: str = "cuda",
        max_frames: int = 16,
        max_pixels: int = 128 * 32 * 32,
        image_patch_size: int = 16,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; CPU inference is not supported in this wrapper.")
        self.device = "cuda"
        device_map = "auto" if device == "auto" else self.device

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype="auto",
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        # Silence unsupported sampling flags.
        if hasattr(self.model, "generation_config"):
            cfg = self.model.generation_config
            if hasattr(cfg, "top_p"):
                cfg.top_p = None
            if hasattr(cfg, "top_k"):
                cfg.top_k = None
            self.model.generation_config = cfg

        self.max_frames = max_frames
        self.max_pixels = max_pixels
        self.image_patch_size = image_patch_size
        print(f"[INFO: Opt-R1] Qwen3HF initialized with model_id={model_id}, device={self.device}")

    def _maybe_download_video(self, video: str) -> (str, Optional[str]):
        if not video.startswith("http"):
            return video, None
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        with requests.get(video, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return path, path

    def _build_messages(self, prompt: str, video: Optional[str], image: Optional[str]) -> List[Dict[str, Any]]:
        if (video is None and image is None) or (video is not None and image is not None):
            raise ValueError("Provide exactly one of `video` or `image`.")
        vision_content: Dict[str, Any]
        if video:
            vision_content = {
                "type": "video",
                "video": video,
                "max_pixels": self.max_pixels,
                "max_frames": self.max_frames,
            }
        else:
            vision_content = {
                "type": "image",
                "image": image,
                "max_pixels": self.max_pixels,
            }
        return [{"role": "user", "content": [vision_content, {"type": "text", "text": prompt}]}]

    def generate(
        self,
        prompt: str,
        *,
        video: Optional[str] = None,
        image: Optional[str] = None,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.7,
    ) -> str:
        cleanup = None
        if video:
            video, cleanup = self._maybe_download_video(video)

        messages = self._build_messages(prompt, video, image)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages],
            return_video_kwargs=True,
            image_patch_size=self.image_patch_size,
            return_video_metadata=True,
        )

        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            **video_kwargs,
            do_resize=False,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
        )
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)]
        output = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        if cleanup:
            try:
                os.remove(cleanup)
            except OSError:
                pass
        return output


if __name__ == "__main__":
    qwen = Qwen3HF()
    default_video = "https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4"
    default_image = "https://ultralytics.com/images/bus.jpg"

    print(qwen.generate("Describe this media.", video=default_video))
    print(qwen.generate("Describe this media.", image=default_image))
