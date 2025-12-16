from swift.llm import PtEngine, InferRequest, RequestConfig
import os

os.environ['MAX_PIXELS'] = '409600' #'1003520'
os.environ['VIDEO_MAX_PIXELS'] = '50176'
os.environ['FPS_MAX_FRAMES'] = '16'

class Qwen3VL(object):
    def __init__(self, model_id='Qwen/Qwen3-VL-2B-Instruct'):
        #self.engine = PtEngine(model_id, attn_impl='flash_attention_2')
        self.engine = PtEngine(model_id)
        self.request_config = RequestConfig(max_tokens=128, temperature=0)
        # Print Init Info
        print(f"[INFO: Opt-R1] Qwen3VL initialized with model_id: {model_id}")

    def generate(self, text, video=None, image=None):
        # video or image
        # video can be a path, url
        # image can be a path, url
        if video is not None:
            infer_request = InferRequest(
                messages=[{
                    "role": "user",
                    "content": "<video>" + text,
                }],
                videos=[video],
            )
        elif image is not None:
            infer_request = InferRequest(
                messages=[{
                    "role": "user",
                    "content": "<image>" + text,
                }],
                images=[image],
            )
        resp_list = self.engine.infer([infer_request], request_config=self.request_config)
        response = resp_list[0].choices[0].message.content
        return response

    def generate_text(self, text):
        infer_request = InferRequest(
                messages=[{
                    "role": "user",
                    "content": text,
                }],
            )
        resp_list = self.engine.infer([infer_request], request_config=self.request_config)
        response = resp_list[0].choices[0].message.content
        return response


# test
if __name__ == "__main__":
    qwen3vl = Qwen3VL()
    default_video = 'https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4'
    default_image = 'https://ultralytics.com/images/bus.jpg'
    response = qwen3vl.generate("Describe this media.", video=default_video)
    print(response)
    response = qwen3vl.generate("Describe this media.", image=default_image)
    print(response)
