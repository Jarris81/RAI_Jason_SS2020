import util.primitive as prim
from transitions import Machine
from util.tower import Tower
import libry as ry
import copy
from functools import partial
from guppy import hpy
import numpy as np


#
MAX_DISTANCE_TOP_GRASP = 1.0  # 1
MAX_GRIPPER_WIDTH = 0.2


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
class TowerBuilder:

    def __init__(self, C, S, V, tau):
        # add all states
        self.grav_comp = prim.GravComp(C, S, V, tau, 1000, vis=False)
        self.top_grasp = prim.TopGrasp(C, S, V, tau, 200, interpolation=True, vis=False)
        self.top_place = prim.TopPlace(C, S, V, tau, 200, interpolation=True, vis=False)
        self.pull_in = prim.PullIn(C, S, V, tau, 200, interpolation=True, vis=False)
        self.push_to_edge = prim.PushToEdge(C, S, V, tau, 600, interpolation=True, vis=False)
        self.edge_grasp = prim.EdgeGrasp(C, S, V, tau, 500, interpolation=True, vis=False)
        self.edge_place = prim.EdgePlace(C, S, V, tau, 300, interpolation=False, vis=True)
        self.edge_drop = prim.AngleEdgePlace(C, S, V, tau, 500, interpolation=True, vis=False)
        self.reset = prim.Drop(C, S, V, tau, 150, interpolation=True, vis=False)
        self.tau = tau
        self.name = "Panda"
        self.state = None
        self.t = 0
        self.goal = None
        self.gripper = "R_gripper"

        self.C = C
        self.V = V
        self.observed_blocks = []

        self.tower = Tower(C, V, [0.0, -0.3, .68])

        # define list of states
        self.states = [self.grav_comp, self.top_grasp, self.top_place, self.edge_grasp, self.edge_drop]

        # create finite state machine
        self.fsm = Machine(states=self.states, initial=self.grav_comp,
                           auto_transitions=False, after_state_change=self.init_state)
        # add all transitions
        self.fsm.add_transition("do_top_grasp", source=self.grav_comp, dest=self.top_grasp,
                                conditions=[self._has_Goal, self._do_top_grasp])
        # edge grasp
        self.fsm.add_transition("do_edge_grasp", source=self.grav_comp, dest=self.edge_grasp,
                                conditions=[self._has_Goal, self._do_edge_grasp])
        self.fsm.add_transition("do_edge_drop", source=self.edge_grasp, dest=self.edge_drop,
                                conditions=[self._is_grasping])
        self.fsm.add_transition("is_done", source=self.top_grasp, dest=self.top_place, conditions=self._is_grasping)
        self.fsm.add_transition("is_done", source=self.top_place, dest=self.grav_comp, conditions=self._is_open,
                                before=self.place_goal_in_tower)
        self.init_state()

        self.cheat = True

    def init_state(self):
        print("inting new state")
        print(self.fsm.state)
        self.fsm.get_state(self.fsm.state).create_primitive(self.t, gripper=self.gripper, goal=self.goal,
                                                        move_to=self.tower.get_placement())

    def step(self, t):
        self.t = t

        # get all possible triggers and do transition if condition fulfills
        for trigger in self.fsm.get_triggers(self.fsm.state):
            # leave if transition was made
            print(trigger)
            if self.fsm.trigger(trigger):
                break  # leave loop when trigger was made
        # make a step with the current state
        self.fsm.get_state(self.fsm.state).step(self.t)

    def _is_grasping(self):
        return self.fsm.get_state(self.fsm.state).is_grasping()

    def _is_open(self):
        return self.fsm.get_state(self.fsm.state).is_open()

    def _is_done(self):
        return self.fsm.get_state(self.fsm.state).is_done(self.t)

    def _has_Goal(self):
        """
        This function should define the next block, which should be placed in the tower
        :param hasGoal:
        :return:
        """
        # get all the blocks which are not in tower
        unplaced_blocks = list(filter(lambda x: self._filter_block(x), self.observed_blocks))

        # return false if no block available to place
        if not len(unplaced_blocks):
            return False

        # would have to sort and filter blocks somehow, according to size, distance etc
        unplaced_blocks.sort(key=self._get_block_utility, reverse=True)  # highest utility should be placed first

        # get the first in list
        self.goal = unplaced_blocks[0]
        print(f"New Goal is : {self.goal}!")
        return True

    def place_goal_in_tower(self):
        print(f"Block {self.goal} was placed!!!")
        self.C.frame(self.goal).setColor([0, 0, 1])
        self.V.setConfiguration(self.C)
        self.V.recopyMeshes(self.C)
        self.tower.add_block(self.goal)
        self.goal = None

    def set_blocks(self, blocks):

        self.observed_blocks = blocks

    def _filter_block(self, block):

        return block not in self.tower.get_blocks()

    def _get_block_utility(self, block):

        block_size = self.C.frame(block).getSize()

        return block_size[0] * block_size[1]

    def _do_top_grasp(self):
        """
        Check if the robot should do a top grasp on the decided goal
        :return: True if a top grasp is possible
        """
        # check if we have a goal
        if not self.goal:
            return False

        # TODO check if block is not too far away

        # check if length or width of block is smaller than gripper
        goal_size = self.C.frame(self.goal).getSize()
        if np.all(goal_size[:2] > MAX_GRIPPER_WIDTH):
            return False

        return True

    def _do_edge_grasp(self):
        """
        Check if the robot should do a top grasp on the decided goal
        :return: True if a top grasp is possible
        """
        # check if we have a goal
        if not self.goal:
            return False

        table_xy_limit = np.array([0.85, 0.08, 1.3, 0.12])
        # check if block is located at edge
        # check if height of block is small enough
        goal_position = self.C.frame(self.goal).getPosition()
        # check if lower and upper limit is ok
        if not np.all(table_xy_limit[:2] < goal_position[:2]) and \
                not np.all(goal_position[:2] < table_xy_limit[2:]):
            return False

        # check if height of block is small enough
        goal_size = self.C.frame(self.goal).getSize()
        if np.all(goal_size[2] > MAX_GRIPPER_WIDTH):
            return False

        return True


# noinspection PyTypeChecker
class EdgeGrasper(TowerBuilder):

    def __init__(self, C, S, V, tau):
        # add all states
        self.grav_comp = prim.GravComp(C, S, V, tau, 1000, vis=False)
        self.pull_in = prim.PullIn(C, S, V, tau, 200, interpolation=True, vis=False)
        self.push_to_edge = prim.PushToEdge(C, S, V, tau, 600, interpolation=True, vis=False)
        self.edge_grasp = prim.EdgeGrasp(C, S, V, tau, 400, interpolation=True, vis=False)
        self.edge_place = prim.EdgePlace(C, S, V, tau, 300, interpolation=False, vis=True)
        self.edge_drop = prim.AngleEdgePlace(C, S, V, tau, 300, interpolation=True, vis=False)
        self.reset = prim.Drop(C, S, V, tau, 150, interpolation=True, vis=False)
        self.tau = tau
        self.name = "Panda"
        self.state = None
        self.t = 0
        self.goal = None
        self.gripper = "R_gripper"

        self.C = C
        self.V = V
        self.observed_blocks = []

        self.tower = Tower(C, V, [0.0, -0.3, .68])

        # define list of states
        self.states = [self.grav_comp, self.pull_in, self.push_to_edge, self.edge_grasp,
                       self.edge_place, self.edge_drop, self.reset]

        # create finite state machine
        self.fsm = Machine(states=self.states, initial=self.grav_comp,
                           auto_transitions=False, after_state_change=self.init_state)
        # add all transitions
        self.fsm.add_transition("pull_in", source=self.grav_comp, dest=self.pull_in, conditions=self._has_Goal)
        self.fsm.add_transition("push_to_edge", source=self.pull_in, dest=self.push_to_edge, conditions=self._is_done)
        self.fsm.add_transition("edge_grasp", source=self.push_to_edge, dest=self.edge_grasp, conditions=self._is_done)
        self.fsm.add_transition("edge_place", source=self.edge_grasp, dest=self.edge_drop, conditions=self._is_grasping)
        self.fsm.add_transition("is_done", source=[self.edge_place], dest=self.grav_comp,
                                conditions=self._is_open,  before=self.place_goal_in_tower)
        self.fsm.add_transition("reset", source=self.edge_drop, dest=self.reset,
                                conditions=self._is_open)
        self.fsm.add_transition("is_done_dropping", source=[self.reset], dest=self.grav_comp,
                                conditions=self._is_done, after=self.place_goal_in_tower)
        self.init_state()

        self.cheat = True


