"""Native TorchRL EnvBase adapter used by BenchMARL, batched entirely on device."""
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import Composite,Unbounded,Categorical
from benchmarl.environments.common import TaskClass
from .env import HomeBatch,OBS

class TorchRLHome(EnvBase):
    def __init__(self,n=256,device='cuda',seed=0,shared=True):
        super().__init__(device=device,batch_size=[n])
        self.sim=HomeBatch(n,device);self.sim.gen.manual_seed(seed or 0);self.shared=shared
        self.observation_spec=Composite(agents=Composite(observation=Unbounded(shape=(n,4,OBS),device=device),shape=(n,4),device=device),shape=(n,),device=device)
        self.action_spec=Composite(agents=Composite(action=Categorical(9,shape=(n,4),device=device),shape=(n,4),device=device),shape=(n,),device=device)
        self.reward_spec=Composite(agents=Composite(reward=Unbounded(shape=(n,4,1),device=device),shape=(n,4),device=device),shape=(n,),device=device)
        self.done_spec=Composite(done=Categorical(2,shape=(n,1),dtype=torch.bool,device=device),terminated=Categorical(2,shape=(n,1),dtype=torch.bool,device=device),shape=(n,),device=device)
    def _set_seed(self,seed):self.sim.gen.manual_seed(seed)
    def _reset(self,tensordict=None,**kwargs):
        # All 96-step episodes are synchronized, collector resets the full batch.
        self.sim.reset();return self._observation()
    def _observation(self):return TensorDict({'agents':TensorDict({'observation':self.sim.observe(self.shared)},batch_size=[self.sim.n,4])},batch_size=self.batch_size,device=self.device)
    def _step(self,td):
        _,r,done,info=self.sim.step(td['agents','action'])
        result=self._observation();result['agents','reward']=r[:,None,None].expand(-1,4,1)/5
        # Finite-horizon daily objective: terminal at midnight, no bootstrap.
        result['done']=torch.full((*self.batch_size,1),done,device=self.device,dtype=torch.bool)
        result['terminated']=result['done'].clone()
        return result

class HomeTask(TaskClass):
    def get_env_fun(self,num_envs,continuous_actions,seed,device):return lambda:TorchRLHome(num_envs,device,seed,self.config.get('shared',True))
    def supports_continuous_actions(self):return False
    def supports_discrete_actions(self):return True
    def has_render(self,env):return False
    def max_steps(self,env):return 96
    def group_map(self,env):return {'agents':['living_room','bedroom','kitchen','bathroom']}
    def state_spec(self,env):return None
    def action_mask_spec(self,env):return None
    def observation_spec(self,env):return env.full_observation_spec_unbatched.clone()
    def action_spec(self,env):return env.full_action_spec_unbatched.clone()
    def info_spec(self,env):return None
    @staticmethod
    def env_name():return 'smarthome'
