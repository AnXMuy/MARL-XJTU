"""Run the reproducible five-seed CUDA training sweep, then paired test evaluation."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from traffic_marl.utils.provenance import metadata,save_json

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parent

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default=str(ROOT/'runs/formal'))
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--seeds',type=int,default=5)
    p.add_argument('--demand-seeds',type=int,default=20)
    p.add_argument('--skip-training',action='store_true')
    p.add_argument('--skip-evaluation',action='store_true')
    args=p.parse_args(); out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    python=Path(sys.executable); env_script=ROOT/'scripts/sumo_env.sh'
    manifest={'status':'running','started_at':'2026-11-19','episodes_per_seed':args.episodes,
        'training_seeds':list(range(args.seeds)),'demand_seeds':list(range(10000,10000+args.demand_seeds)),
        'algorithms':['mappo','ippo','idqn'],'device':'cuda','provenance':metadata(),'commands':[]}
    save_json(out/'formal_manifest.json',manifest)
    if not args.skip_training:
        for algorithm in manifest['algorithms']:
            for seed in manifest['training_seeds']:
                run=out/'training'/algorithm/f'seed_{seed}'; latest=run/'latest.pt'
                cmd=[str(env_script),str(python),str(ROOT/'scripts/train.py'),'--algorithm',algorithm,
                    '--seed',str(seed),'--episodes',str(args.episodes),'--validate-every','10',
                    '--randomize-demand','--device','cuda','--config',str(ROOT/'configs/train.json'),
                    '--output',str(run)]
                if latest.exists(): cmd.append('--resume')
                manifest['commands'].append({'algorithm':algorithm,'training_seed':seed,'cmd':cmd})
                save_json(out/'formal_manifest.json',manifest)
                print(json.dumps({'event':'training_start','algorithm':algorithm,'seed':seed,'device':'cuda'}),flush=True)
                result=subprocess.run(cmd,cwd=WORKSPACE)
                if result.returncode:
                    manifest.setdefault('failures',[]).append({'algorithm':algorithm,'seed':seed,'returncode':result.returncode})
                    save_json(out/'formal_manifest.json',manifest)
                    raise SystemExit(result.returncode)
    if not args.skip_evaluation:
        checkpoints=[]
        for algorithm in manifest['algorithms']:
            for seed in manifest['training_seeds']:
                path=out/'training'/algorithm/f'seed_{seed}'/'best.pt'
                if not path.exists(): raise FileNotFoundError(path)
                checkpoints.append(f'{algorithm}:{seed}={path}')
        cmd=[str(env_script),str(python),str(ROOT/'scripts/run_matrix.py'),
             '--output',str(out/'evaluation'),'--seeds',str(args.seeds),
             '--demand-seeds',str(args.demand_seeds),'--duration','3600','--size','3']
        for checkpoint in checkpoints: cmd.extend(['--checkpoint-dir',checkpoint])
        manifest['evaluation_command']=cmd; save_json(out/'formal_manifest.json',manifest)
        result=subprocess.run(cmd,cwd=WORKSPACE)
        if result.returncode: raise SystemExit(result.returncode)
    manifest['status']='complete'; save_json(out/'formal_manifest.json',manifest)
if __name__=='__main__': main()
