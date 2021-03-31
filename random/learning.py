import acme
from acme.utils import loggers

class RandomLearner(acme.Learner):

    def __init__(self, logger=None):
        self._logger = loggers.make_default_logger('learner')
        pass

    def step(self):
        """Run learning step. Nothing to do 
        for random agent."""
        pass

    def get_variables(self, names):
        """No variables to return for random agent"""
        return None

if __name__=="__main__":

    learner = RandomLearner()
    learner.step()