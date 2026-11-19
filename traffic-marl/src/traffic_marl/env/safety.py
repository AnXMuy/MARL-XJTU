from dataclasses import dataclass

@dataclass
class Signal:
    min_green: int = 10
    yellow: int = 3
    all_red: int = 2
    phase: int = 0
    stage: str = 'green'
    elapsed: int = 0
    switches: int = 0
    def mask(self):
        return [True, self.stage == 'green' and self.elapsed >= self.min_green]
    def request(self, action):
        if action not in (0,1) or not self.mask()[action]:
            raise ValueError('illegal signal action')
        if action:
            self.stage='yellow'
            self.elapsed=0
            self.switches+=1
    def tick(self):
        self.elapsed+=1
        if self.stage=='yellow' and self.elapsed>=self.yellow:
            self.stage='red'
            self.elapsed=0
        elif self.stage=='red' and self.elapsed>=self.all_red:
            self.phase=1-self.phase
            self.stage='green'
            self.elapsed=0
    def state(self, groups):
        if self.stage=='red':
            return 'r'*len(groups)
        return ''.join(('G' if self.stage=='green' else 'y') if g==self.phase else 'r' for g in groups)
