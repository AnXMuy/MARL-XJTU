import hashlib
from pathlib import Path
from openai import OpenAI

class TTSClient:
    def __init__(self, url="http://localhost:8091/v1", model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", cache=None):
        self.client=OpenAI(base_url=url,api_key="local",timeout=30,max_retries=0); self.model=model
        self.cache=Path(cache or Path(__file__).resolve().parents[1]/"assets/tts_cache"); self.cache.mkdir(parents=True,exist_ok=True)
    def speak(self,text,voice="vivian"):
        path=self.cache/(hashlib.sha1((text+voice+self.model).encode()).hexdigest()+".wav")
        try:
            if not path.exists():
                response=self.client.audio.speech.create(model=self.model, input=text, voice=voice, response_format="wav")
                temporary=path.with_suffix(".tmp"); response.stream_to_file(temporary); temporary.replace(path)
            return str(path)
        except Exception: return None
