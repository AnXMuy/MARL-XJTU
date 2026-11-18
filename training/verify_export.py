"""Check exported greedy inference against the actual BenchMARL policy."""
import argparse,tempfile,json
import torch
from tensordict import TensorDict
from benchmarl.algorithms import MappoConfig,IppoConfig
from benchmarl.experiment import Experiment,ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from experiments.v2.torchrl_env import HomeTask
from experiments.v2.env import HomeBatch
from experiments.v2.evaluate import Learned

def main():
 p=argparse.ArgumentParser();p.add_argument('checkpoint');a=p.parse_args();torch.set_num_threads(1)
 data=torch.load(a.checkpoint,weights_only=False,map_location='cpu');args=data['args']
 c=ExperimentConfig.get_from_yaml();c.sampling_device=c.train_device=c.buffer_device='cpu'
 c.share_policy_params=True;c.prefer_continuous_actions=False;c.max_n_frames=768;c.max_n_iters=None
 c.on_policy_n_envs_per_worker=8;c.on_policy_collected_frames_per_batch=768;c.on_policy_minibatch_size=768
 c.loggers=[];c.evaluation=False;c.render=False;c.checkpoint_at_end=False;c.checkpoint_interval=0
 actor=MlpConfig.get_from_yaml();actor.num_cells=[args['hidden']]*args['layers'];actor.activation_class=torch.nn.Tanh
 critic=MlpConfig.get_from_yaml();critic.num_cells=[args['hidden']*2]*args['layers'];critic.activation_class=torch.nn.Tanh
 with tempfile.TemporaryDirectory() as tmp:
  c.save_folder=tmp
  exp=Experiment(task=HomeTask('four_rooms',{'shared':True}),algorithm_config=(MappoConfig if args['algorithm']=='mappo' else IppoConfig).get_from_yaml(),model_config=actor,critic_model_config=critic,seed=42,config=c)
  exp.policy.load_state_dict(data['policy']);e=HomeBatch(8);e.reset(seed=775)
  policy=Learned(a.checkpoint)
  from torchrl.envs.utils import set_exploration_type,ExplorationType
  with torch.no_grad(),set_exploration_type(ExplorationType.DETERMINISTIC):
   for t in range(96):
    td=TensorDict({'agents':TensorDict({'observation':e.observe()},batch_size=[8,4])},batch_size=[8])
    actual=exp.policy(td)['agents','action'];exported=policy.act(e)
    assert torch.equal(actual,exported),(t,actual,exported)
    e.step(exported)
 print(json.dumps({'passed':True,'comparisons':96*8*4,'checkpoint':a.checkpoint}))
if __name__=='__main__':main()
