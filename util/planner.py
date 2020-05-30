import numpy as np
import libry as ry


def check_if_goal_constant(goals, minumum_goals=2, tolerance=0.005):
    if len(goals) < minumum_goals:
        return False
    checker = goals[0]
    for goal in goals:
        if goal.shape[0] != checker.shape[0] or goal.shape[1] != checker.shape[1]:
            return False
        elif not np.isclose(goal, checker, atol=tolerance).all():
            return False
    return True

    # score = np.vstack(goals[:minumum_goals]) - goals[0]
    # return np.isclose(score, np.zeros_like(score), atol=tolerance).all()


def set_goal_ball(C, V, position, radius):
    goal = C.frame("goal")
    goal.setShape(ry.ST.sphere, [radius+0.005])
    goal.setColor([0, 1, 0])
    goal.setPosition(position)
    goal.setContact(1)
    V.setConfiguration(C)
