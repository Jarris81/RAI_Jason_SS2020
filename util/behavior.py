import util.primitive as prim
import util.constants as const
from transitions.extensions import GraphMachine as Machine
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

class PickNPlace:

    def __init__(self, C, S, V, tau):
        # add all states

        # primitives
        self.grav_comp = prim.GravComp(C, S, V, tau, 10, vis=False)
        self.reset = prim.Reset(C, S, V, tau, 1, komo=False, vis=False)
        self.top_grasp = prim.TopGrasp(C, S, V, tau, 1, komo=False, vis=False)
        self.top_place = prim.TopPlace(C, S, V, tau, 1, komo=False, vis=False)

        # functions
        self.tau = tau
        self.name = "Panda"
        self.state = None
        self.t = 0
        self.goal = None
        self.active_gripper = None
        self.next_gripper = None

        self.C = C
        self.V = V
        self.observed_blocks = []
        self.tower = Tower(C, V, [0.0, -.4, .7])

        # define list of states
        self.states = [self.grav_comp, self.top_grasp, self.top_place, self.reset]

        # create finite state machine
        self.fsm = Machine(states=self.states, initial=self.grav_comp,
                           auto_transitions=False, after_state_change=self.init_state)
        # RESET
        self.fsm.add_transition("reset", source=self.grav_comp, dest=self.reset,
                                conditions=[self._gripper_changed])
        # self.fsm.add_transition("reset_done", source=self.reset, dest=self.grav_comp,
        #                         conditions=[self._is_done], after=self._set_new_gripper)

        # TOP GRASP
        self.fsm.add_transition("do_top_grasp", source=self.grav_comp, dest=self.top_grasp,
                                conditions=[self._has_Goal, self._do_top_grasp])
        self.fsm.add_transition("do_top_place", source=self.top_grasp, dest=self.top_place,
                                conditions=self._is_grasping)

        # transitions returning to grav comp after placing block
        self.fsm.add_transition("is_done_placing", source=[self.top_place],
                                dest=self.grav_comp, conditions=self._is_open, after=self.place_goal_in_tower)
        # initiate initial state
        self.init_state()

        self.fsm.get_graph().draw('simple_pick_and_place.png', prog="dot", )

        self.cheat = True

    def init_state(self):
        print("----- Initiating new State -----")
        self.fsm.get_state(self.fsm.state).set_min_overhead(self.tower.get_placement()[2] + const.MIN_OVERHEAD)
        self.fsm.get_state(self.fsm.state).create_primitive(self.t, gripper=self.active_gripper, goal=self.goal,
                                                            move_to=self.tower.get_placement())

    def step(self, t):
        self.t = t
        # update tower
        self.tower.update()
        # get all possible triggers and do transition if condition fulfills
        for trigger in self.fsm.get_triggers(self.fsm.state):
            # leave if transition was made
            if self.fsm.trigger(trigger):
                print("----- Triggering transition:", trigger, " -----")
                break  # leave loop when trigger was made
        # make a step with the current state
        self.fsm.get_state(self.fsm.state).step(self.t)

    def _gripper_changed(self):
        return self.next_gripper is not self.active_gripper

    def _set_new_gripper(self):
        self.active_gripper = self.next_gripper

    def _is_grasping(self):
        return self.fsm.get_state(self.fsm.state).is_grasping()

    def _is_open(self):
        return self.fsm.get_state(self.fsm.state).is_open()

    def _is_done(self):
        return self.fsm.get_state(self.fsm.state).is_done(self.t)

    def _is_first_block(self):
        return len(self.tower.get_blocks()) == 0

    def _is_not_first_block(self):
        return not self._is_first_block()

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
            self.next_gripper = None
            return False

        # would have to sort and filter blocks somehow, according to size, distance etc
        unplaced_blocks.sort(key=self._get_block_utility, reverse=True)  # highest utility should be placed first

        # get the first in list
        self.goal = unplaced_blocks[0]
        # check which gripper to use for the primitive
        goal_posx = self.C.frame(self.goal).getPosition()[0]
        if goal_posx <= 0:
            self.next_gripper = const.PANDA_L_GRIPPER
        else:
            self.next_gripper = const.PANDA_R_GRIPPER
        # check if an active gripper was set
        if self.active_gripper is None:
            self.active_gripper = self.next_gripper
            return True
        # check if we need to reset first
        if self.active_gripper is not self.next_gripper:
            return False
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
        """
        Filter all blocks which are already in tower and are too far away
        :param block:
        :return:
        """
        return block not in self.tower.get_blocks()

    def _get_block_utility(self, block):
        """
        Sort Blocks after utlity, atm only size of XY-Area
        :param block:
        :return:
        """
        block_size = self.C.frame(block).getSize()

        return block_size[0] * block_size[1]

    def _do_top_grasp(self):
        """
        Conditional Function for state machine
        Check if the robot should do a top grasp on the decided goal
        :return: True if a top grasp is possible
        """
        # check if we have a goal
        if not self.goal:
            return False
        # TODO: check if block is not too far away
        # check if length or width of block is smaller than gripper
        goal_size = self.C.frame(self.goal).getSize()
        if np.all(goal_size[:2] > MAX_GRIPPER_WIDTH):
            return False

        return True

    def _do_pull_in(self):
        """
        Conditional Function for state machine
        Check if the robot should do a top grasp on the decided goal
        :return: True if a top grasp is possible
        """
        # check if we have a goal
        if not self.goal:
            return False
        goal_position = self.C.frame(self.goal).getPosition()
        push_to_edge_y_limits = np.array([-0.1, 0.2])
        if not push_to_edge_y_limits[0] < goal_position[1] < push_to_edge_y_limits[-1]:
            return False

        # check if length or width of block is smaller than gripper
        goal_size = self.C.frame(self.goal).getSize()
        if np.all(goal_size[2] > MAX_GRIPPER_WIDTH):
            return False

        return True

    def _do_push_to_edge(self):
        """
        Conditional Function for state machine
        Check if the robot should do a top grasp on the decided goal
        :return: True if a top grasp is possible
        """
        # check if we have a goal
        if not self.goal:
            return False
        goal_position = self.C.frame(self.goal).getPosition()
        push_to_edge_y_limits = np.array([-0.1, 0.2])
        if not push_to_edge_y_limits[0] < goal_position[1] < push_to_edge_y_limits[-1]:
            return False

        # check if length or width of block is smaller than gripper
        goal_size = self.C.frame(self.goal).getSize()
        if np.all(goal_size[2] > MAX_GRIPPER_WIDTH):
            return False

        return True

    def _do_edge_grasp(self):
        """
        Conditional Function for state machine
        Check if the robot should do a edge grasp on the decided goal
        :return: True if a edge grasp is possible
        """
        # check if we have a goal
        if not self.goal:
            return False
        # TODO: check if block is not too far away
        table_right_edge_xy_limit = np.array([0.85, -0.15, 1.0, 0.12])
        table_left_edge_xy_limit = np.array([-1.0, -0.15, -0.85, 0.12])
        # check if block is located at edge
        goal_position = self.C.frame(self.goal).getPosition()
        is_right_edge = np.all(table_right_edge_xy_limit[:2] < goal_position[:2]) and \
                        np.all(goal_position[:2] < table_right_edge_xy_limit[2:])
        is_left_edge = np.all(table_left_edge_xy_limit[:2] < goal_position[:2]) and \
                       np.all(goal_position[:2] < table_left_edge_xy_limit[2:])
        # check if lower and upper limit is ok
        if not is_left_edge and not is_right_edge:
            return False

        # check if height of block is small enough
        goal_size = self.C.frame(self.goal).getSize()
        if np.all(goal_size[2] > MAX_GRIPPER_WIDTH):
            return False

        return True


# noinspection PyTypeChecker
class TowerBuilder:

    def __init__(self, C, S, V, tau):
        # add all states

        # primitives
        self.grav_comp = prim.GravComp(C, S, V, tau, 10, vis=False)
        self.reset = prim.Reset(C, S, V, tau, 1, komo=False, vis=False)
        self.top_grasp = prim.TopGrasp(C, S, V, tau, 1, komo=False, vis=False)
        self.top_place = prim.TopPlace(C, S, V, tau, 1, komo=False, vis=False)
        self.pull_in = prim.PullIn(C, S, V, tau, 2, komo=False, vis=False)
        self.push_to_edge = prim.PushToEdge(C, S, V, tau, 3, komo=False, vis=False)
        self.edge_grasp = prim.EdgeGrasp(C, S, V, tau, 2, komo=False, vis=False)
        self.edge_place = prim.EdgePlace(C, S, V, tau, 4, komo=False, vis=False)
        self.edge_drop = prim.AngleEdgePlace(C, S, V, tau, 5, komo=False, vis=False)
        self.drop = prim.Drop(C, S, V, tau, 3, komo=False, vis=False)
        self.move_away = prim.MoveAway(C, S, V, tau, 3, komo=False, vis=False)

        # functions
        self.tau = tau
        self.name = "Panda"
        self.state = None
        self.t = 0
        self.goal = None
        self.active_gripper = None
        self.next_gripper = None

        self.C = C
        self.V = V
        self.observed_blocks = []
        self.tower = Tower(C, V, [0.0, -.4, .7])

        # define list of states
        self.states = [self.grav_comp, self.reset, self.top_grasp, self.top_place, self.edge_grasp, self.edge_drop,
                       self.drop, self.push_to_edge, self.edge_place, self.move_away, self.pull_in]

        m = Model()
        # create finite state machine
        self.fsm = Machine(model=m, states=self.states, initial=self.grav_comp,
                           auto_transitions=False, after_state_change=self.init_state, use_pygraphviz=True,
                           title='Tower Builder State Machine', show_conditions=True)
        # RESET
        self.fsm.add_transition("gripper_changed", source=self.grav_comp, dest=self.reset,
                                conditions=[self._gripper_changed])
        self.fsm.add_transition("reset_done", source=self.reset, dest=self.grav_comp,
                                conditions=[self._is_done], after=self._set_new_gripper)

        # TOP GRASP
        self.fsm.add_transition("do_top_grasp", source=self.grav_comp, dest=self.top_grasp,
                                conditions=[self._has_Goal, self._do_top_grasp])
        self.fsm.add_transition("do_top_place", source=self.top_grasp, dest=self.top_place,
                                conditions=self._is_grasping)

        # Primitives for EDGE GRASP
        self.fsm.add_transition("do_edge_grasp", source=self.grav_comp, dest=self.edge_grasp,
                                conditions=[self._has_Goal, self._do_edge_grasp])
        self.fsm.add_transition("do_edge_drop", source=self.edge_grasp, dest=self.edge_place,
                                conditions=[self._is_grasping, self._is_not_first_block])
        self.fsm.add_transition("do_edge_drop", source=self.edge_grasp, dest=self.edge_drop,
                                conditions=[self._is_grasping, self._is_first_block])
        self.fsm.add_transition("do_edge_drop", source=[self.edge_place], dest=self.move_away,
                                conditions=[self._is_open])
        self.fsm.add_transition("do_edge_drop", source=[self.edge_drop], dest=self.drop,
                                conditions=[self._is_open])
        # PUSH To Edge
        self.fsm.add_transition("do_push_to_edge", source=self.grav_comp, dest=self.push_to_edge,
                                conditions=[self._has_Goal, self._do_push_to_edge])
        self.fsm.add_transition("done_push", source=self.push_to_edge, dest=self.grav_comp, conditions=self._is_done)

        # Pull in
        self.fsm.add_transition("do_pull_in", source=self.grav_comp, dest=self.pull_in,
                                conditions=[self._has_Goal, self._do_pull_in])
        self.fsm.add_transition("done_pull_in", source=self.pull_in, dest=self.grav_comp, conditions=self._is_done)

        # transitions returning to grav comp after placing block
        self.fsm.add_transition("is_done_placing", source=[self.top_place],
                                dest=self.grav_comp, conditions=self._is_open, after=self.place_goal_in_tower)
        self.fsm.add_transition("is_done_dropping", source=[self.drop, self.move_away],
                                dest=self.grav_comp, conditions=self._is_done, after=self.place_goal_in_tower)

        # transitions returning to grav comp, after not placing a block
        self.fsm.add_transition("done_push", source=self.push_to_edge, dest=self.grav_comp, conditions=self._is_done)

        # transitions when something went wrong
        self.fsm.add_transition("grasp_failed", source=[self.top_grasp, self.edge_grasp],
                               dest=self.reset, conditions=self._grasp_failed, after=self.place_goal_in_tower)

        # initiate initial state
        #self.init_state()

        # draw the whole graph ...
        m.get_graph().draw('my_state_diagram.png', prog="dot", )
        m.show
        # ... or just the region of interest
        # (previous state, active state and all reachable states)
        #roi = m.get_graph(show_roi=True).draw('my_state_diagram.png', prog="dot")

        self.cheat = True

    def init_state(self):
        print("----- Initiating new State -----")
        self.fsm.get_state(self.fsm.state).set_min_overhead(self.tower.get_placement()[2] + const.MIN_OVERHEAD)
        self.fsm.get_state(self.fsm.state).create_primitive(self.t, gripper=self.active_gripper, goal=self.goal,
                                                            move_to=self.tower.get_placement())

    def step(self, t):
        self.t = t
        # update tower
        self.tower.update()
        # get all possible triggers and do transition if condition fulfills
        for trigger in self.fsm.get_triggers(self.fsm.state):
            # leave if transition was made
            if self.fsm.trigger(trigger):
                print("----- Triggering transition:", trigger, " -----")
                break  # leave loop when trigger was made
        # make a step with the current state
        self.fsm.get_state(self.fsm.state).step(self.t)

    def _gripper_changed(self):
        return self.next_gripper is not self.active_gripper

    def _set_new_gripper(self):
        self.active_gripper = self.next_gripper

    def _is_grasping(self):
        return self.fsm.get_state(self.fsm.state).is_grasping()

    def _is_open(self):
        return self.fsm.get_state(self.fsm.state).is_open()

    def _is_done(self):
        return self.fsm.get_state(self.fsm.state).is_done(self.t)

    def _is_first_block(self):
        return len(self.tower.get_blocks()) == 0

    def _is_not_first_block(self):
        return not self._is_first_block()

    def _grasp_failed(self):
        return self.fsm.get_state(self.fsm.state).grasp_failed()

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
            self.next_gripper = None
            return False

        # would have to sort and filter blocks somehow, according to size, distance etc
        unplaced_blocks.sort(key=self._get_block_utility, reverse=True)  # highest utility should be placed first

        # get the first in list
        self.goal = unplaced_blocks[0]
        # check which gripper to use for the primitive
        goal_posx = self.C.frame(self.goal).getPosition()[0]
        if goal_posx <= 0:
            self.next_gripper = const.PANDA_L_GRIPPER
        else:
            self.next_gripper = const.PANDA_R_GRIPPER
        # check if an active gripper was set
        if self.active_gripper is None:
            self.active_gripper = self.next_gripper
            return True
        # check if we need to reset first
        if self.active_gripper is not self.next_gripper:
            return False
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
        """
        Filter all blocks which are already in tower and are too far away
        :param block:
        :return:
        """
        return block not in self.tower.get_blocks()

    def _get_block_utility(self, block):
        """
        Sort Blocks after utlity, atm only size of XY-Area
        :param block:
        :return:
        """
        block_size = self.C.frame(block).getSize()

        return block_size[0] * block_size[1]

    def _do_top_grasp(self):
        """
        Conditional Function for state machine
        Check if the robot should do a top grasp on the decided goal
        :return: True if a top grasp is possible
        """
        # check if we have a goal
        if not self.goal:
            return False
        # TODO: check if block is not too far away
        # check if length or width of block is smaller than gripper
        goal_size = self.C.frame(self.goal).getSize()
        if np.all(goal_size[:2] > MAX_GRIPPER_WIDTH):
            return False

        return True

    def _do_pull_in(self):
        """
        Conditional Function for state machine
        Check if the robot should do a top grasp on the decided goal
        :return: True if a top grasp is possible
        """
        # check if we have a goal
        if not self.goal:
            return False
        goal_position = self.C.frame(self.goal).getPosition()
        push_to_edge_y_limits = np.array([-0.1, 0.2])
        if not push_to_edge_y_limits[0] < goal_position[1] < push_to_edge_y_limits[-1]:
            return False

        # check if length or width of block is smaller than gripper
        goal_size = self.C.frame(self.goal).getSize()
        if np.all(goal_size[2] > MAX_GRIPPER_WIDTH):
            return False

        return True

    def _do_push_to_edge(self):
        """
        Conditional Function for state machine
        Check if the robot should do a top grasp on the decided goal
        :return: True if a top grasp is possible
        """
        # check if we have a goal
        if not self.goal:
            return False
        goal_position = self.C.frame(self.goal).getPosition()
        push_to_edge_y_limits = np.array([-0.1, 0.2])
        if not push_to_edge_y_limits[0] < goal_position[1] < push_to_edge_y_limits[-1]:
            return False

        # check if length or width of block is smaller than gripper
        goal_size = self.C.frame(self.goal).getSize()
        if np.all(goal_size[2] > MAX_GRIPPER_WIDTH):
            return False

        return True

    def _do_edge_grasp(self):
        """
        Conditional Function for state machine
        Check if the robot should do a edge grasp on the decided goal
        :return: True if a edge grasp is possible
        """
        # check if we have a goal
        if not self.goal:
            return False
        # TODO: check if block is not too far away
        table_right_edge_xy_limit = np.array([0.85, -0.15, 1.0, 0.12])
        table_left_edge_xy_limit = np.array([-1.0, -0.15, -0.85, 0.12])
        # check if block is located at edge
        goal_position = self.C.frame(self.goal).getPosition()
        is_right_edge = np.all(table_right_edge_xy_limit[:2] < goal_position[:2]) and \
                        np.all(goal_position[:2] < table_right_edge_xy_limit[2:])
        is_left_edge = np.all(table_left_edge_xy_limit[:2] < goal_position[:2]) and \
                       np.all(goal_position[:2] < table_left_edge_xy_limit[2:])
        # check if lower and upper limit is ok
        if not is_left_edge and not is_right_edge:
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
        self.pull_in = prim.PullIn(C, S, V, tau, 200, komo=False, vis=False)
        self.push_to_edge = prim.PushToEdge(C, S, V, tau, 600, komo=False, vis=False)
        self.edge_grasp = prim.EdgeGrasp(C, S, V, tau, 400, komo=False, vis=False)
        self.edge_place = prim.EdgePlace(C, S, V, tau, 300, komo=False, vis=True)
        self.edge_drop = prim.AngleEdgePlace(C, S, V, tau, 300, komo=False, vis=False)
        self.reset = prim.Drop(C, S, V, tau, 150, komo=False, vis=False)
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
                                conditions=self._is_open, before=self.place_goal_in_tower)
        self.fsm.add_transition("reset", source=self.edge_drop, dest=self.reset,
                                conditions=self._is_open)
        self.fsm.add_transition("is_done_dropping", source=[self.reset], dest=self.grav_comp,
                                conditions=self._is_done, after=self.place_goal_in_tower)
        self.init_state()

        self.cheat = True


class Model:

    def __init__(self):
        i =0
