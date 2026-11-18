import base64
import mimetypes
from pathlib import Path
from openai import OpenAI

class ASRClient:
    def __init__(self, url="http://localhost:8000/v1", model="Qwen/Qwen3-ASR-1.7B"):
        self.client = OpenAI(base_url=url, api_key="local", timeout=30, max_retries=0); self.model=model
    def transcribe(self, path):
        data = base64.b64encode(Path(path).read_bytes()).decode()
        mime = mimetypes.guess_type(path)[0] or "audio/wav"
        result = self.client.chat.completions.create(model=self.model,messages=[
            {"role":"system","content":"睡眠模式、离家模式、观影模式、省电、客厅、卧室、厨房、卫生间、温度"},
            {"role":"user","content":[{"type":"input_audio","input_audio":{"data":f"data:{mime};base64,{data}","format":"wav"}}]}])
        return result.choices[0].message.content
