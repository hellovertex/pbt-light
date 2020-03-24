import os
import time

import tensorflow as tf
import numpy as np
from typing import Dict
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver as default_driver
from tf_agents.eval import metric_utils
from tf_agents.policies import policy_saver
from tf_agents.utils import common

from utils import FP


class PBTRunner(object):
    def __init__(self, population, env, num_to_evolve=2):
        self.population = population
        self.env = env
        self._num_to_evolve = num_to_evolve
        self._epoch_counter = 0
        # add driver for each member that collects trajectories
        for member in population.members:
            member.driver = default_driver(env=self.env, policy=member.agent.collect_policy,
                                           observers=[member.replay_buffer.add_batch] +
                                                     member.step_metrics + member.train_metrics,
                                           num_episodes=FP.COLLECT_EPISODES_PER_ITERATION)

        # initialize lazily in run_pbt(...)
        self.global_step = None
        # dirs
        self.root_dir = None
        self.train_dir = None
        self.eval_dir = None
        self.saved_model_dir = None
        # intervals
        self.train_checkpoint_interval = None
        self.policy_checkpoint_interval = None
        self.summary_interval = None
        self.log_interval = None

    def run_training(self, num_environment_steps, use_tf_functions=True):
        """
        Args:
            num_environment_steps: per agent
        """
        # run the collect drivers to fill replay buffer of each model
        for i, member in enumerate(self.population.members):
            print("BEFORE SETTING SUMMARY WRITER")
            member.train_summary_writer.set_as_default()
            print("AFTER SETTING SUMMARY WRITER")# train_step_counter was initially set in get_agent
            # and is manually incremented in ppo_agent.train, as required in ABC tf.TfAgent
            print("BEFORE SETTING CONTEXT MANAGER")
            with tf.compat.v2.summary.record_if(lambda: tf.math.equal(
                    member.agent.train_step_counter % self.summary_interval, 0)
            ):
                print("After SETTING CONTEXT MANAGER")
                def train_step():
                    trajectories = member.replay_buffer.gather_all()
                    return member.agent.train(experience=trajectories)

                if use_tf_functions:
                    # todo: Enable once the cause for slowdown was identified.
                    member.driver.run = common.function(member.driver.run, autograph=False)
                    member.agent.train = common.function(member.agent.train, autograph=False)
                    train_step = common.function(train_step)

                timed_at_step = 0
                while (member.step_metrics[FP.IDX_ENV_STEPS].result().numpy()) < num_environment_steps * self._epoch_counter:
                    collect_time = 0
                    train_time = 0
                    train_step_count = member.agent.train_step_counter.numpy()

                    # Collect
                    start_time = time.time()
                    member.driver.run()  # maybe wrap with common.function
                    collect_time += time.time() - start_time

                    # TRAIN
                    start_time = time.time()
                    total_loss, _ = train_step()
                    member.replay_buffer.clear()
                    train_time += time.time() - start_time

                    # Compute metrics
                    for train_metric in member.train_metrics:
                        train_metric.tf_summaries(
                            train_step=train_step_count, step_metrics=member.step_metrics)

                    # LOGGING
                    if train_step_count % self.log_interval == 0:
                        print(f'Training agent {i} in epoch {self._epoch_counter}')
                        print(f'step = '
                              f'{train_step_count:.2f}, '
                              f'loss = {total_loss:.2f}')
                        steps_per_sec = (train_step_count - timed_at_step) / (collect_time + train_time)
                        print(f'steps/sec = {steps_per_sec:.2f}')
                        print(f'collect_time = {collect_time:.2f}, train_time = {train_time:.2f}')
                        timed_at_step = train_step_count

    def rank_members(self, num_to_evolve):
        # avg_score = member.train_metrics[FP.INDEX_AVG_RETURN_METRIC].result().numpy()
        model_results = [member.train_metrics[FP.INDEX_AVG_RETURN_METRIC].result().numpy()
                         for member in self.population.members]
        sorted_indices = np.argsort(model_results)
        partners = list()
        for bad_index in sorted_indices[:num_to_evolve]:
            assigned_partner = np.random.choice(sorted_indices[num_to_evolve:])
            partners.append((bad_index, assigned_partner))
        return sorted_indices, partners

    def exploit(self, pairs):
        for i_bad, i_good in pairs:
            bad_model = self.population.members[i_bad].agent
            good_model = self.population.members[i_good].agent
            for attr, _ in self.population.hyperparams.items():
                # todo change learning rate inside optimizer inside ppo_agent
                assert hasattr(bad_model, attr) and hasattr(good_model, attr)
                setattr(bad_model, attr, getattr(good_model, attr))

    def explore(self, sorted_indices, mutation_prob=0.3):
        for idx in sorted_indices[:self._num_to_evolve]:
            member = self.population.members[idx]
            for attr, attr_mutation_fn in self.population.hyperparams.items():
                if np.random.uniform() < mutation_prob:
                    mutation = attr_mutation_fn()
                    assert hasattr(member.agent, attr)
                    setattr(member.agent, attr, mutation)

    def run_epoch(self, num_environment_steps):
        self.run_training(num_environment_steps)
        sorted_indices, pairs = self.rank_members(self._num_to_evolve)
        self.exploit(pairs)
        self.explore(sorted_indices)
        self.maybe_save_models()

    def _initialize_checkpointers_and_summary_writers(self, root_dir):
        self.root_dir = root_dir
        self.train_dir = os.path.join(root_dir, 'train')
        self.eval_dir = os.path.join(root_dir, 'eval')
        """ 
            The SavedModel that is exported can be loaded via
            `tf.compat.v2.saved_model.load` (or `tf.saved_model.load` in TF2).  It
            will have available signatures (concrete functions): `action` and
            `get_initial_state`.  
        """
        self.saved_model_dir = os.path.join(root_dir, 'policy_saved_model')

        def _create_checkpointer(ckpt_dir, member, ckpt='train'):
            if ckpt == 'train':
                return common.Checkpointer(ckpt_dir=ckpt_dir,
                                           agent=member.agent,
                                           global_step=member.agent.train_step_counter,
                                           metrics=metric_utils.MetricsGroup(
                                               member.step_metrics + member.train_metrics, 'train_metrics'))
            elif ckpt == 'policy':
                return common.Checkpointer(ckpt_dir=os.path.join(ckpt_dir, 'policy'),
                                           policy=member.agent.policy,
                                           global_step=member.agent.train_step_counter)
            else:
                raise ValueError

        # Each member will write checkpoints and summaries separately and to its own directory
        # This way it will be easier to restore the best models by simply comparing their numbers
        # on Tensorboard and loading from the corresponding directory
        for i, member in enumerate(self.population.members):
            train_dir = os.path.join(os.path.join(self.root_dir, f'agent_{i}'), 'train')
            # todo run eval_metrics eagerly on separate eval env
            # eval_dir = os.path.join(os.path.join(self.root_dir, f'agent_{i}'), 'eval')
            member.train_summary_writer = tf.compat.v2.summary.create_file_writer(train_dir, flush_millis=1000)
            # member.eval_summary_writer = tf.compat.v2.summary.create_file_writer(eval_dir, flush_millis=1000)
            member.train_checkpointer = _create_checkpointer(train_dir, member, ckpt='train')
            member.policy_checkpointer = _create_checkpointer(train_dir, member, ckpt='policy')
            member.saved_model = policy_saver.PolicySaver(member.agent.policy, train_step=member.agent.train_step_counter)
            member.train_checkpointer.initialize_or_restore()

    def maybe_save_models(self):
        """ Helper function that checks for logging intervals """
        for member in self.population.members:
            if member.agent.train_step_counter % self.train_checkpoint_interval == 0:
                member.train_checkpointer.save(global_step=member.agent.train_step_counter)
            if member.agent.train_step_counter % self.policy_checkpoint_interval == 0:
                member.policy_checkpointer.save(global_step=member.agent.train_step_counter)
                saved_model_path = os.path.join(
                    self.saved_model_dir, 'policy_' + ('%d' % member.agent.train_step_counter).zfill(9))
                member.saved_model.save(saved_model_path)

    def run_pbt(self,
                root_dir,
                num_epochs,
                num_env_steps_per_epoch,
                train_checkpoint_interval=500,
                policy_checkpoint_interval=500,
                summary_interval=50,
                log_interval=50):
        """
        Args:
             root_dir: Where /train and /eval will be created that contain Summaries
             num_epochs: number of training epochs
             num_env_steps_per_epoch: steps taken in environment per epoch
             train_checkpoint_interval: number of env steps after which train_checkpoint is created
             policy_checkpoint_interval:  number of env steps after which policy_checkpoint is created
             summary_interval:  number of env steps after which summaries are written for train and eval
             log_interval: number of env steps after which current train results are logged to console
        """
        # Init lazy attrs
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        # set intervals
        self.train_checkpoint_interval = train_checkpoint_interval
        self.policy_checkpoint_interval = policy_checkpoint_interval
        self.summary_interval = summary_interval
        self.log_interval = log_interval
        # init checkpointing and summary writing
        self._initialize_checkpointers_and_summary_writers(root_dir)

        # run population based training
        for i in range(num_epochs):
            self._epoch_counter += 1
            self.run_epoch(num_env_steps_per_epoch)

