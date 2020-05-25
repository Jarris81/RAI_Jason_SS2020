import numpy as np
import libry as ry


def check_if_goal_constant(goals, minumum_goals=2, tolerance=0.005):
    if len(goals) < minumum_goals:
        return False
    return np.isclose(np.vstack(goals), goals[0], atol=tolerance).all()


def set_goal_ball(C, V, position, radius):
    goal = C.frame("goal")
    goal.setShape(ry.ST.sphere, [radius+0.005])
    goal.setColor([0, 1, 0])
    goal.setPosition(position)
    goal.setContact(1)
    V.setConfiguration(C)
