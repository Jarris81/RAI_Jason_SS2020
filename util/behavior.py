import util.primitive as prim
from transitions import Machine
from util.tower import Tower
import libry as ry
import copy
from functools import partial
from guppy import hpy


class GrabAndLift:

    def __init__(self, C, S, V, tau):
        # add all states
        self.grav_comp = prim.GravComp(C, S, V, tau, 1000, gripper="R_gripper")
        self.side_grasp = prim.SideGrasp(C, S, V, tau, 100, gripper="R_gripper")
        self.lift_up = prim.LiftUp(C, S, V, tau, 100, gripper="R_gripper")
        self.name = "Panda"
        self.state = None
        self.t = 0
        self.hasGoal = False

        # define list of states
        self.states = [self.side_grasp, self.lift_up, self.grav_comp]

        # create finite state machine
        self.fsm = Machine(states=self.states, initial=self.grav_comp, auto_transitions=False)
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
        self.grav_comp = prim.GravComp(C, S, V, tau, 1000, vis=False)
        self.top_grasp = prim.TopGrasp(C, S, V, tau, 100, interpolation=True, vis=False)
        self.top_place = prim.TopPlace(C, S, V, tau, 100, interpolation=True, vis=False)
        self.tau = tau
        self.name = "Panda"
        self.state = None
        self.t = 0
        self.goal = None

        self.C = C
        self.V = V
        self.observed_blocks = []

        self.tower = Tower(C, V, [0.0, -0.3, .68])

        # define list of states
        self.states = [self.top_grasp, self.top_place, self.grav_comp]

        # create finite state machine
        self.fsm = Machine(states=self.states, initial=self.grav_comp,
                           auto_transitions=False)
        # add all transitions
        self.fsm.add_transition("do_top_grasp", source=self.grav_comp, dest=self.top_grasp, conditions=self._has_Goal,
                                after=self.init_state)
        self.fsm.add_transition("is_done", source=self.top_grasp, dest=self.top_place, conditions=self._is_grasping,
                                after=self.init_state)
        self.fsm.add_transition("is_done", source=self.top_place, dest=self.grav_comp, conditions=self._is_open,
                                before=self.place_goal_in_tower, after=self.init_state)
        self.init_state()

        self.cheat = True

    def init_state(self):
        print("inting new state")
        print(self.fsm.state)
        self.fsm.get_state(self.fsm.state).create_primitive(self.t, gripper="R_gripper", goal=self.goal,
                                                        move_to=self.tower.get_placement())

    def step(self, t):
        self.t = t

        # get all possible triggers and do transition if condition fulfills
        for trigger in self.fsm.get_triggers(self.fsm.state):
            # leave if transition was made
            if self.fsm.trigger(trigger):
                break
        # make a step with the current state
        self.fsm.get_state(self.fsm.state).step(self.t)

    def _is_grasping(self):
        return self.fsm.get_state(self.fsm.state).is_grasping()

    def _is_open(self):
        return self.fsm.get_state(self.fsm.state).is_open()

    def _has_Goal(self):
        """
        This function should define the next block, which should be placed in the tower
        :param hasGoal:
        :return:
        """

        # get all the blocks which are not in tower
        unplaced_blocks = [block for block in self.observed_blocks if block not in self.tower.get_blocks()]
        # return false if no block available to place
        if not len(unplaced_blocks):
            return False

        # would have to sort and filter blocks somehow, according to size, distance etc
        sorted_filtered_blocks = unplaced_blocks

        if not len(sorted_filtered_blocks):
            return False
        # get the first in list
        self.goal = sorted_filtered_blocks[0]
        print(f"New Goal is : {self.goal}!")
        return True

    def place_goal_in_tower(self):
        print(f"Block {self.goal} was placed!!!")
        self.C.frame(self.goal).setColor([0, 0, 1])
        self.V.recopyMeshes(self.C)
        self.tower.add_block(self.goal)
        self.goal = None

    def set_blocks(self, blocks):

        self.observed_blocks = blocks

