
class Member(object):
    def __init__(self, agent, replay_buffer, train_metrics):
        """ Each Member consists of agent, driver, replay-buffer and metrics """
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.train_metrics = train_metrics  # assumed to contain step_metrics as well

        # set lazily in PBTRunner __init__(...)
        self._driver = None
        # set lazily in PBTRunner run_pbt(...)
        self._train_checkpointer = None  # tf_agents.utils.common.Checkpointer
        self._policy_checkpointer = None
        self._train_summary_writer = None  # tf.summary.SummaryWriter
        self._eval_summary_writer = None
        self._saved_model = None  # tf_agents.policies.PolicySaver:

    """ ---------------------------------------- """
    """ Boilerplate for lazily set attributes  """
    """ ---------------------------------------- """

    @property
    def driver(self):
        return self._driver

    @driver.setter
    def driver(self, value):
        if self._driver is not None:
            # Possibly throw an Error here
            print(f'Member {self} already has a driver')
        self._driver = value

    @property
    def train_checkpointer(self):
        return self._train_checkpointer

    @train_checkpointer.setter
    def train_checkpointer(self, value):
        if self._train_checkpointer is not None:
            # Possibly throw an Error here
            print(f'Member {self} already has a train_checkpointer')
        self._train_checkpointer = value

    @property
    def policy_checkpointer(self):
        return self._policy_checkpointer

    @policy_checkpointer.setter
    def policy_checkpointer(self, value):
        if self._policy_checkpointer is not None:
            # Possibly throw an Error here
            print(f'Member {self} already has a policy_checkpointer')
        self._policy_checkpointer = value

    @property
    def train_summary_writer(self):
        return self._train_summary_writer

    @train_summary_writer.setter
    def train_summary_writer(self, value):
        if self._train_summary_writer is not None:
            print(f'Member {self} already has a train_summary_writer')
        self._train_summary_writer = value

    @property
    def eval_summary_writer(self):
        return self._eval_summary_writer

    @eval_summary_writer.setter
    def eval_summary_writer(self, value):
        if self._eval_summary_writer is not None:
            print(f'Member {self} already has a_eval_summary_writer')
        self._eval_summary_writer = value

    @property
    def saved_model(self):
        return self._saved_model

    @saved_model.setter
    def saved_model(self, value):
        if self._saved_model is not None:
            print(f'Member {self} already has a saved_model')
        self._saved_model = value

