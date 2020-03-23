import tensorflow as tf
import numpy as np
# tf_agents imports
from tf_agents.environments import tf_py_environment, parallel_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network, value_network
from environments import test_env
# project imports
from agents.ppo import ppo_agent


class FP(object):
    """ Param Container so that we can fold them in Pycharm for readability """
    # Training
    NUM_PARALLEL_ENVS = 25
    COLLECT_EPISODES_PER_ITERATION = 30
    # get-functions
    ACTOR_FC_LAYERS = (200, 100)
    VALUE_FC_LAYERS = (200, 100)
    REPLAY_BUFFER_CAPACITY = 1000
    # ppo agent
    PPO_NUM_EPOCHS = 25
    PPO_HYPERPARAMS = {# 'learning_rate': lambda: np.random.uniform(1e-5, 1e-3),
        'num_epochs': lambda: np.random.randint(2, 36),
        'importance_ratio_clipping': lambda: np.random.uniform(0.05, 0.4),
        'entropy_regularization': lambda: np.random.uniform(0.001, 1),
        'value_pred_loss_coef': lambda: np.random.uniform(0.1, 1.7),
        'kl_cutoff_factor': lambda: np.random.uniform(0.5, 2.5),
        'kl_cutoff_coef': lambda: np.random.uniform(1., 2000.),
        # 'nsteps': lambda: np.random.randint(2, 40)
    }
    # Indices of train_metrics
    INDEX_NUM_EPISODES_METRIC = 0
    IDX_ENV_STEPS = 1
    INDEX_AVG_RETURN_METRIC = 0


def get_tf_env():
    def _load_env():
        return test_env.CountingEnv(steps_per_episode=10)
    tf_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment(
        [lambda: _load_env()] * FP.NUM_PARALLEL_ENVS))
    return tf_env


def get_networks(tf_env, actor_fc_layers, value_fc_layers):
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        fc_layer_params=actor_fc_layers)
    value_net = value_network.ValueNetwork(
        tf_env.observation_spec(),
        fc_layer_params=value_fc_layers)
    return actor_net, value_net


def get_tf_ppo_agent(tf_env, actor_net, value_net, member_id, num_epochs=25, learning_rate=1e-3):
    tf_agent = ppo_agent.PPOAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        member_id=member_id,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
        actor_net=actor_net,
        value_net=value_net,
        entropy_regularization=0.1,
        importance_ratio_clipping=0.2,
        discount_factor=1.,
        lambda_value=1.,
        normalize_observations=False,
        normalize_rewards=False,
        use_gae=True,
        num_epochs=num_epochs,
        debug_summaries=False,
        summarize_grads_and_vars=False,
        train_step_counter=None)
    tf_agent.initialize()  # check if this is necessary
    return tf_agent


def get_replay_buffer(data_spec):
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=data_spec,
        batch_size=FP.NUM_PARALLEL_ENVS,
        max_length=FP.REPLAY_BUFFER_CAPACITY)


def get_metrics():
    step_metrics = [tf_metrics.NumberOfEpisodes(), tf_metrics.EnvironmentSteps()]

    return step_metrics, [tf_metrics.AverageReturnMetric(batch_size=FP.NUM_PARALLEL_ENVS),
                           tf_metrics.AverageEpisodeLengthMetric(batch_size=FP.NUM_PARALLEL_ENVS)]
