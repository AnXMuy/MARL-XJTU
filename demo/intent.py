"""Validated intents. Language models can change preferences, never device actions."""
import json
import re
import numpy as np
ROOMS = ('living_room','bedroom','kitchen','bathroom')
SCENES = ('none','sleep','away','movie','home')
def default_preference():
    return np.array([0,0,0,0,1,.5,1,1,.4,1,0,0,0,0],dtype=np.float32)

ALIASES = {"客厅": "living_room", "卧室": "bedroom", "厨房": "kitchen", "卫生间": "bathroom", "浴室": "bathroom"}
CLARIFY = "您是想调温度、切换场景，还是查询状态？"

def scene_preference(scene):
    p = default_preference()
    p[9:] = 0; p[9 + SCENES.index(scene)] = 1
    if scene == "sleep": p[:4] = [-3, -2, -3, -3]; p[4:8] = [0, .5, 0, 0]
    elif scene == "away": p[:4] = -3; p[4:8] = 0; p[8] = .9
    elif scene == "movie": p[4] = .5
    return p

def validate(data):
    if not isinstance(data, dict) or data.get("intent") not in ("set_preference", "scene", "query", "undo"):
        raise ValueError("Invalid intent")
    rooms = data.get("rooms", [])
    if not isinstance(rooms, list) or any(r not in ROOMS for r in rooms): raise ValueError("Invalid room")
    result = {"intent": data["intent"], "rooms": rooms}
    for key, lo, hi in (("delta_temp", -3, 3), ("brightness", 0, 1), ("w_eco", 0, 1)):
        value = data.get(key)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"Invalid {key}")
            if key == "brightness" and value not in (0, .5, 1): raise ValueError("Invalid brightness")
        result[key] = value
    result["scene"] = data.get("scene")
    if result["scene"] not in (None, *SCENES): raise ValueError("Invalid scene")
    if result["intent"] == "scene" and result["scene"] is None: raise ValueError("Missing scene")
    if result["intent"] == "set_preference" and all(result[k] is None for k in ("delta_temp", "brightness", "w_eco")):
        raise ValueError("Empty update")
    result["answer_hint"] = None
    return result

def rule_intent(text):
    rooms = list(dict.fromkeys(v for k, v in ALIASES.items() if k in text))
    if any(k in text for k in ("全屋", "家里", "整个家")): rooms = list(ROOMS)
    base = {"intent": "set_preference", "rooms": rooms}
    if any(k in text for k in ("撤销", "恢复刚才", "上一步")): base["intent"] = "undo"
    elif any(k in text for k in ("多少度", "几度", "现在温度", "电费多少")): base["intent"] = "query"
    elif any(k in text for k in ("睡", "晚安")): base.update(intent="scene", scene="sleep")
    elif any(k in text for k in ("离家", "出门", "拜拜")): base.update(intent="scene", scene="away")
    elif any(k in text for k in ("观影", "电影")): base.update(intent="scene", scene="movie")
    elif any(k in text for k in ("回家", "到家")): base.update(intent="scene", scene="home")
    elif any(k in text for k in ("省电", "电费太", "省着", "节能")): base["w_eco"] = .9
    elif any(k in text for k in ("冷", "冻", "暖", "升温", "调高")): base["delta_temp"] = 2
    elif any(k in text for k in ("热", "凉", "降温", "调低")): base["delta_temp"] = -2
    elif "灯" in text: base["brightness"] = 0 if "关" in text else .5 if "暗" in text else 1
    else: return None
    return validate(base)

class IntentParser:
    def __init__(self, client=None): self.client = client
    def parse(self, text, current):
        if self.client:
            try:
                raw = self.client.complete(text, current)
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
                return validate(json.loads(raw)), "llm"
            except (Exception,): pass
        return rule_intent(text), "rule"

def merge(preference, intent, occupied):
    if intent["intent"] == "scene": return scene_preference(intent["scene"])
    p = preference.copy()
    rooms = intent["rooms"] or [ROOMS[i] for i, present in enumerate(occupied) if present] or ["living_room"]
    for room in rooms:
        i = ROOMS.index(room)
        if intent.get("delta_temp") is not None: p[i] = np.clip(p[i] + intent["delta_temp"], -3, 3)
        if intent.get("brightness") is not None: p[4 + i] = intent["brightness"]
    if intent.get("w_eco") is not None: p[8] = intent["w_eco"]
    return p
