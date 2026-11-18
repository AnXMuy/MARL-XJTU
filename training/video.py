"""Render full-residence comparison from recorded final-test actions, not a script."""
import json,os
from pathlib import Path
import torch
import numpy as np
import imageio.v2 as imageio
from playwright.sync_api import sync_playwright
from experiments.v2.env import HomeBatch
from demo.residence import render

def main():
 d=json.loads(Path('assets/final_study.json').read_text());out=Path('assets/demo');out.mkdir(exist_ok=True,parents=True)
 labels=['mappo_lookahead','coordinated_rule'];titles=['MAPPO + 前瞻调控','协调规则 · 同条件对照']
 runs=[d['results']['changes'][p]['days'][0] for p in labels]
 homes=[HomeBatch(),HomeBatch()]
 for e,run in zip(homes,runs):e.reset(seed=run['seed'],switches=False)
 with sync_playwright() as p:
  browser=p.chromium.launch(executable_path=os.environ.get('CHROMIUM_PATH'),headless=True,args=['--no-sandbox'])
  page=browser.new_page(viewport={'width':1800,'height':1020},device_scale_factor=1)
  with imageio.get_writer(out/'paired_day.mp4',fps=8,codec='libx264',macro_block_size=1) as writer:
   for t in range(96):
    boards=[]
    for e,run,title in zip(homes,runs,titles):
     rows=run['trajectory'][:t+1];r=rows[-1];e.t=t+1
     for attr in ('temps','hvac','lights'):setattr(e,attr,torch.tensor([r[attr]]))
     e.targets=lambda r=r:(torch.tensor([r['target']]),torch.tensor([r['lights']]),torch.tensor([r['active']]))
     board=render(e,rows,title)
     cost=sum(z['cost'] for z in rows);comfort=sum(z['degree_hours'] for z in rows)
     boards.append(f'<section>{board}<p>累计电费 ¥{cost:.2f}　·　温度偏离 {comfort:.2f} °C·h</p></section>')
    page.set_content('<html><meta charset="UTF-8"><style>body{margin:0;padding:24px;background:#f1f0e9;color:#234b3e;font-family:sans-serif}h1{font-size:28px}main{display:flex;gap:24px}section{width:50%;background:#e8e8dd;padding:18px;border-radius:12px}svg{width:100%;height:740px}.board-top{display:flex;justify-content:space-between}p{font-size:23px}</style><h1>同一住宅，同一天，同一需求 · '+f'{(t+1)//4:02d}:{(t+1)%4*15:02d}</h1><main>'+''.join(boards)+'</main><p style="font-size:15px">真实留出测试日 100020 · 07:00 回家升温 / 17:00 省电 / 21:00 睡眠 · 仿真结果</p></html>')
    import io
    writer.append_data(imageio.imread(io.BytesIO(page.screenshot()),format='png')[:,:,:3])
  browser.close()
 print(out/'paired_day.mp4')
if __name__=='__main__':main()
