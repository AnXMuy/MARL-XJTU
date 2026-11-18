"""Main training entrypoint: actual BenchMARL Experiment + TorchRL MAPPO/IPPO."""
import argparse,json,time
from pathlib import Path
import torch
from benchmarl.algorithms import MappoConfig,IppoConfig
from benchmarl.experiment import Experiment,ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from .torchrl_env import HomeTask
from .env import VERSION,HomeBatch
from benchmarl.experiment.callback import Callback
from .evaluate import Learned

class Selection(Callback):
    def __init__(self,path,args):
        super().__init__();self.path=path;self.args=args;self.best=-float('inf');self.history=[]
    def on_evaluation_end(self,rollouts):
        data={'policy':self.experiment.policy.state_dict(),'args':vars(self.args),'environment':VERSION}
        torch.save(data,self.path/'last_actor.pt')
        controller=Learned(self.path/'last_actor.pt',self.args.device)
        e=HomeBatch(128,self.args.device);e.reset(seed=900000,switches=True);score=0
        with torch.no_grad():
            for _ in range(96):
                _,r,_,_=e.step(controller.act(e));score+=r.mean()
        score=float(score);row={'frames':self.experiment.total_frames,'validation_return':score,'gpu_peak_mib':torch.cuda.max_memory_allocated()/2**20}
        self.history.append(row);print('VALIDATION',json.dumps(row),flush=True)
        (self.path/'validation.json').write_text(json.dumps(self.history,indent=2))
        if score>self.best:self.best=score;torch.save(data,self.path/'best_actor.pt')

def main():
    p=argparse.ArgumentParser();p.add_argument('--algorithm',choices=['mappo','ippo'],default='mappo');p.add_argument('--steps',type=int,default=10000000)
    p.add_argument('--envs',type=int,default=1024);p.add_argument('--hidden',type=int,default=256);p.add_argument('--layers',type=int,default=3)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda');p.add_argument('--output',default='results/benchmarl/mappo42')
    args=p.parse_args();torch.set_num_threads(4)
    c=ExperimentConfig.get_from_yaml();c.sampling_device=args.device;c.train_device=args.device;c.buffer_device=args.device
    c.share_policy_params=True;c.prefer_continuous_actions=False;c.lr=3e-4;c.adam_eps=1e-5;c.clip_grad_val=.5
    c.max_n_frames=args.steps;c.max_n_iters=None;c.on_policy_collected_frames_per_batch=args.envs*96
    c.on_policy_n_envs_per_worker=args.envs;c.on_policy_n_minibatch_iters=6;c.on_policy_minibatch_size=8192
    c.evaluation=True;c.render=False;c.evaluation_episodes=32;c.evaluation_interval=args.envs*96*5;c.evaluation_static=True
    c.loggers=['csv'];c.create_json=False;c.save_folder=str(Path(args.output).resolve());c.checkpoint_at_end=True;c.checkpoint_interval=args.envs*96*10
    a=(MappoConfig if args.algorithm=='mappo' else IppoConfig).get_from_yaml();a.entropy_coef=.01;a.critic_coef=.5;a.clip_epsilon=.2;a.lmbda=.95
    actor=MlpConfig.get_from_yaml();actor.num_cells=[args.hidden]*args.layers;actor.activation_class=torch.nn.Tanh
    critic=MlpConfig.get_from_yaml();critic.num_cells=[args.hidden*2]*args.layers;critic.activation_class=torch.nn.Tanh
    path=Path(args.output);path.mkdir(parents=True,exist_ok=True)
    (path/'run_manifest.json').write_text(json.dumps({'environment':VERSION,'framework':'BenchMARL/TorchRL','args':vars(args),'reward_scale':.2,'notes':'Actual BenchMARL runner, collector, GAE and ClipPPOLoss. Nine categorical joint room actions.'},indent=2))
    exp=Experiment(task=HomeTask('four_rooms',{'shared':True}),algorithm_config=a,model_config=actor,critic_model_config=critic,seed=args.seed,config=c,callbacks=[Selection(path,args)])
    start=time.time();exp.run()
    torch.save({'policy':exp.policy.state_dict(),'args':vars(args),'environment':VERSION},path/'actor_export.pt')
    (path/'throughput.json').write_text(json.dumps({'seconds':time.time()-start,'frames':exp.total_frames,'gpu_peak_mib':torch.cuda.max_memory_allocated()/2**20},indent=2))
if __name__=='__main__':main()
