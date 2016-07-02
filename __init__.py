from gym.envs.registration import registry, register, make, spec

from bandits import MultiArmedBanditEnv
from bandits import MultiArmedBanditRegretEnv
from bandits import ContextualBanditEnv

from classification import MNISTEnv

register(
    id='MultiArmedBandit-v0',
    entry_point='ranch:MultiArmedBanditEnv',
    timestep_limit=1000
)

register(
    id='MultiArmedBanditRegret-v0',
    entry_point='ranch:MultiArmedBanditRegretEnv',
    timestep_limit=1000
)

register(
    id='ContextualBandit-v0',
    entry_point='ranch:ContextualBanditEnv',
    timestep_limit=3000
)

register(
    id='MNIST-v0',
    entry_point='ranch:MNISTEnv',
    timestep_limit=1000
)