from gym.envs.registration import registry, register, make, spec

from bandits import FiveArmedBanditEnv, EightArmedBanditEnv, TwentyArmedBanditEnv
from bandits import FiveArmedBanditLateRewardEnv, EightArmedBanditLateRewardEnv, TwentyArmedBanditLateRewardEnv
from bandits import FiveArmedBanditRegretEnv, EightArmedBanditRegretEnv, TwentyArmedBanditRegretEnv
from bandits import FiveArmedBanditRegretLateEnv, EightArmedBanditRegretLateEnv, TwentyArmedBanditRegretLateEnv
from bandits import ContextualBanditEnv

from classification import MNISTEnv

register(
    id='EightArmedBandit-v0',
    entry_point='ranch:EightArmedBanditEnv',
    timestep_limit=2000
)

register(
    id='TwentyArmedBandit-v0',
    entry_point='ranch:TwentyArmedBanditEnv',
    timestep_limit=2000
)

register(
    id='EightArmedBanditRegret-v0',
    entry_point='ranch:EightArmedBanditRegretEnv',
    timestep_limit=2000
)

register(
    id='TwentyArmedBanditRegret-v0',
    entry_point='ranch:TwentyArmedBanditRegretEnv',
    timestep_limit=2000
)

register(
    id='EightArmedBanditLateReward-v0',
    entry_point='ranch:EightArmedBanditLateRewardEnv',
    timestep_limit=2000
)

register(
    id='TwentyArmedBanditLateReward-v0',
    entry_point='ranch:TwentyArmedBanditLateRewardEnv',
    timestep_limit=2000
)

register(
    id='EightArmedBanditRegretLate-v0',
    entry_point='ranch:EightArmedBanditRegretLateEnv',
    timestep_limit=2000
)

register(
    id='TwentyArmedBanditRegretLate-v0',
    entry_point='ranch:TwentyArmedBanditRegretLateEnv',
    timestep_limit=2000
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