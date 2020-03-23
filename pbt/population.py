from utils import get_tf_env, get_networks, get_tf_ppo_agent, get_replay_buffer, get_train_metrics
from utils import FP
from typing import Dict
import member


def create_members(num_members, tf_env=None):
    """ Can go inside Population class with more parametrization options depending on use-case. """
    if tf_env is None:
        tf_env = get_tf_env()
    members = list()
    for i in range(num_members):
        actor_net, value_net = get_networks(tf_env, FP.ACTOR_FC_LAYERS, FP.VALUE_FC_LAYERS)
        agent = get_tf_ppo_agent(tf_env, actor_net, value_net, member_id=i, num_epochs=FP.PPO_NUM_EPOCHS)
        replay_buffer = get_replay_buffer(agent.collect_data_spec)
        train_metrics = get_train_metrics()
        members.append(member.Member(agent, replay_buffer, train_metrics))
    return members


class Population(object):
    def __init__(self, members, hyperparams: Dict):
        self.members = members
        self.size = len(members)
        self.hyperparams = hyperparams  # todo exploit networks, environment
