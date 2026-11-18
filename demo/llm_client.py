import json
from openai import OpenAI

class LLMClient:
    def __init__(self, url="http://localhost:8002/v1", model="Qwen/Qwen3-8B"):
        self.client = OpenAI(base_url=url, api_key="local", timeout=12, max_retries=0)
        self.model = model
    def complete(self, text, current):
        prompt = """你是智能家居意图解析器，只输出 JSON。禁止输出设备动作。字段：intent(set_preference|scene|query|undo),rooms(living_room|bedroom|kitchen|bathroom数组),delta_temp(-3至3相对增量或null),brightness(0|0.5|1|null),w_eco(0至1或null),scene(none|sleep|away|movie|home|null),answer_hint(null)。未提及字段为null。例：我有点冷=>{"intent":"set_preference","rooms":[],"delta_temp":2}；我要睡觉了=>{"intent":"scene","rooms":[],"scene":"sleep"}；客厅多少度=>{"intent":"query","rooms":["living_room"]}。当前偏好："""
        out = self.client.chat.completions.create(model=self.model, messages=[{"role":"system","content":prompt + json.dumps(list(map(float,current)))},{"role":"user","content":text}], temperature=0, max_tokens=300, extra_body={"chat_template_kwargs":{"enable_thinking":False}})
        return out.choices[0].message.content
