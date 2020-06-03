import util.primitive as prim
from transitions import Machine
from functools import partial


class GrabAndLift:

    def __init__(self, C, S, tau, V=None):
        # add all states
        self.grav_comp = prim.GravComp(C, S, tau, 1000, gripper="R_gripper", V=V)
        self.side_grasp = prim.SideGrasp(C, S, tau, 100, gripper="R_gripper", V=V)
        self.lift_up = prim.LiftUp(C, S, tau, 100, gripper="R_gripper", V=V)
        self.name = "Panda"
        self.state = None
        self.t = 0
        self.hasGoal = False

        # define list of states
        self.states = [self.side_grasp, self.lift_up, self.grav_comp]

        # create finite state machine
        self.fsm = Machine(model=self, states=self.states, initial=self.grav_comp, auto_transitions=False)
        # add all transitions
        self.fsm.add_transition("is_done", source="grav_comp", dest="side_grasp", conditions="get_has_goal")
        self.fsm.add_transition("is_done", source="side_grasp", dest="lift_up", conditions=self._is_grasping)
        self.init_state()

    # def on_enter_lift_up(self):
    #      self.lift_up.get_komo()
    #      print(self.state)

    def init_state(self):
        self.fsm.get_state(self.state).create_primitive(self.t)

    def step(self, t):
        self.t = t

        # get all possible triggers and do transition if condition fulfills
        for trigger in self.fsm.get_triggers(self.state):
            # leave if transition was made
            if self.trigger(trigger):
                break

        # make a step with the current state
        self.fsm.get_state(self.state).step(self.t)

    def _is_grasping(self):
        return self.fsm.get_state(self.state).is_grasping()

    def set_Goal(self, hasGoal):
        self.hasGoal = hasGoal

    def get_has_goal(self):
        return self.hasGoal


# noinspection PyTypeChecker
class PickAndPlace:

    def __init__(self, C, S, V, tau):
        # add all states
        self.grav_comp = prim.GravComp(C, S, V, tau, 1000, gripper="R_gripper", vis=False)
        self.top_grasp = prim.TopGrasp(C, S, V, tau, 100, gripper="R_gripper", vis=False)
        self.top_place = prim.TopPlace(C, S, V, tau, 100, gripper="R_gripper", vis=False)
        self.name = "Panda"
        self.state = None
        self.t = 0
        self.hasGoal = False

        self.tower_position = [0.0, 0.3, .68]

        # define list of states
        self.states = [self.top_grasp, self.top_place, self.grav_comp]

        # create finite state machine
        self.fsm = Machine(model=self, states=self.states, initial=self.grav_comp, auto_transitions=False)
        # add all transitions
        self.fsm.add_transition("do_top_grasp", source=self.grav_comp, dest=self.top_grasp, conditions="get_has_goal")
        self.fsm.add_transition("is_done", source=self.top_grasp, dest=self.top_place, conditions=self._is_grasping)
        self.fsm.add_transition("is_done", source=self.top_place, dest=self.grav_comp, conditions=self._is_open)
        self.init_state()

        self.cheat = True

    # def on_enter_lift_up(self):
    #      self.lift_up.get_komo()
    #      print(self.state)

    def init_state(self):
        self.fsm.get_state(self.state).create_primitive(self.t, move_to=self.tower_position)

    def step(self, t):
        self.t = t

        # get all possible triggers and do transition if condition fulfills
        for trigger in self.fsm.get_triggers(self.state):
            # leave if transition was made
            if self.trigger(trigger):
                break

        # make a step with the current state
        self.fsm.get_state(self.state).step(self.t)

    def _is_grasping(self):
        return self.fsm.get_state(self.state).is_grasping()

    def _is_open(self):
        return self.fsm.get_state(self.state).is_open()

    def set_Goal(self, hasGoal):
        self.hasGoal = hasGoal

    def get_has_goal(self):
        return self.hasGoal