from dataclasses import asdict, dataclass, field
import hashlib
import json

@dataclass
class Config:
    size: int = 3
    spacing: int = 300
    duration: int = 3600
    warmup: int = 300
    demand_end: int | None = None
    rate: float = 300.0  # veh/h per boundary entrance
    entrance_rates: dict[str, float] = field(default_factory=dict)
    scenario: str = 'uniform'
    seed: int = 10000
    decision: int = 5
    min_green: int = 10
    max_green: int = 60
    yellow: int = 3
    all_red: int = 2
    fixed_green: int = 30
    local_reward: bool = False
    downstream_obs: bool = True
    queue_only: bool = False
    neighbors: bool = False
    def __post_init__(self):
        if self.size not in (1, 3, 4):
            raise ValueError('size must be 1, 3 or 4')
        if not 0 <= self.warmup < self.duration <= 10800:
            raise ValueError('require 0 <= warmup < duration <= 10800')
        if self.scenario not in ('uniform', 'morning', 'evening', 'pulse'):
            raise ValueError('unknown demand scenario')
        if not 0 <= self.rate <= 3000 or any(not 0 <= r <= 3000 for r in self.entrance_rates.values()):
            raise ValueError('rates must be between 0 and 3000 veh/h')
        if self.decision < 1 or self.min_green < self.decision or self.max_green < self.min_green:
            raise ValueError('invalid green/decision timing')
        if self.yellow < 1 or self.all_red < 1 or self.fixed_green < self.min_green:
            raise ValueError('invalid clearance/fixed timing')
        if self.demand_end is not None and not 0 < self.demand_end <= self.duration:
            raise ValueError('invalid demand_end')
        valid = {f'{s}{i}' for s in 'WENS' for i in range(self.size)}
        if set(self.entrance_rates) - valid:
            raise ValueError('unknown entrance')
    def dict(self):
        return asdict(self)
    def digest(self):
        return hashlib.sha256(json.dumps(self.dict(), sort_keys=True).encode()).hexdigest()
