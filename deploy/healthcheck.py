import json
import urllib.request

SERVICES={"ASR":8000,"TTS":8091,"LLM":8002}
def check():
    result={}
    for name,port in SERVICES.items():
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/v1/models",timeout=.6) as response:
                result[name]=response.status==200 and bool(json.load(response).get("data"))
        except Exception: result[name]=False
    return result
if __name__=="__main__": print(json.dumps(check(),ensure_ascii=False,indent=2))
