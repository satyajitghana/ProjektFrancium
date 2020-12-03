from typing import Tuple, Optional

from francium.core import BaseEnvironment, State
import numpy as np


class Environment(BaseEnvironment):
    def __init__(self, x_bounds: Tuple[float, float], y_bounds: Tuple[float, float], goal_val: Optional[State] = None,
                 tolerance: Optional[float] = 1e-4):

        BaseEnvironment.__init__(self, x_bounds, y_bounds, goal_val, tolerance)

    def evaluate_state(self, state: State) -> Tuple[float, bool]:
        eval_val = self.evaluation_func(state['x'], state['y'])

        if self.goal_val:
            is_done = True if np.abs(eval_val - self.goal_val['z']) < self.tolerance else False
        else:
            is_done = False

        return eval_val, is_done