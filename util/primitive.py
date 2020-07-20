import libry as ry
import numpy as np
import time
from transitions import State
import util.bezier as beziers
import util.constants as const

import util.constants as constants


class Primitive(State):

    def __init__(self, name, C, S, V, tau, n_steps,
                 grasping=False, holding=False, placing=False,
                 komo=False, vis=False):

        State.__init__(self, name)
        self.duration = tau * n_steps
        self.n_steps = n_steps*0.001/tau
        self.tau = tau
        self.C = C
        self.S = S
        self.V = V
        self.grasping = grasping
        self.holding = holding
        self.placing = placing
        self.vis = vis
        self.use_komo = komo
        self.max_place_counter = 20
        self.min_overhead = const.MIN_OVERHEAD

        # mask to make sure the fingers do not change
        self.mask_gripper = np.asarray([0] * 16)
        self.mask_gripper[-1] = 1
        self.mask_gripper[7] = 1

        # x_position of table edges
        self.left_edge_x = -1
        self.right_edge_x = 1

        # these attribute need to be set in the method create_primitive
        self.gripper = None
        self.goal = None
        self.t_start = None
        self.start_config = None
        self.goal_config = None
        self.initial_goal_position = None
        self.q_start = None
        self.q_goal = None
        self.iK = None
        self.komo = None
        self.q_values = None
        self.place_counter = None
        self.is_in_world = None
        self.needs_overhead_start = None

    def create_primitive(self, t_start, gripper, goal, move_to=None):
        self.t_start = t_start
        self.gripper = gripper
        self.goal = goal
        self.q_start = self.C.getJointState()
        self.start_config = self.C.getFrameState()
        # need to find out if we need overhead
        self.needs_overhead_start = self.min_overhead > self.C.frame(gripper).getPosition()[2]

        if not self.goal is None:
            self.initial_goal_position = self.C.frame(self.goal).getPosition()
        # get the goal configuration
        self.goal_config = self._get_goal_config(move_to)
        self.C.setFrameState(self.goal_config)

        self.place_counter = 0
        self.is_in_world = False

        # visualize goal config if V is set
        if self.vis:
            print(f"Displaying Goal Config of Primitive: {self.name}")
            self.V.setConfiguration(self.C)
            time.sleep(5)
        # reset initial config in configuration space
        self.q_goal = self.C.getJointState()
        self.C.setFrameState(self.start_config)
        if self.use_komo:
            # get the komo path for the primitive and optimize
            self.komo = self._get_komo(move_to)
            self.komo.optimize(False)
            # visualize komo path if V is set
            if self.vis:
                V2 = self.komo.view()
                time.sleep(2)
                V2.playVideo()
                time.sleep(2)
        else:
            self.q_values = []
            phases, bezier_profiles, q_points = self._get_joint_interpolation(move_to=move_to)
            # show via points
            if self.vis:
                for i, q_via in enumerate(q_points):
                    self.C.setJointState(q_via)
                    self.V.setConfiguration(self.C)
                    print("showing via point ", i)
                    time.sleep(1)

            # create movements from values
            for i, (phase, bezier_profile) in enumerate(zip(phases, bezier_profiles)):
                delta = q_points[i + 1] - q_points[i]
                # create n steps between 0 and 1
                steps = np.linspace(0, 1, int(phase * self.n_steps))
                # create bezier
                bezier = beziers.create_bezier(bezier_profile)
                # return list of values
                self.q_values.extend([q_points[i] + delta * bezier.solve(t) for t in steps])

    def set_min_overhead(self, min_overhead):
        self.min_overhead = min_overhead

    def _get_goal_config(self, move_to=None):
        print("Method: :get_goal_config not implemented for Primtive: ", __name__)
        return

    def _get_komo(self, move_to=None):
        print("Method: :get_komo not implemented for Primtive: ", __name__)
        return

    def _get_joint_interpolation(self, move_to=None):
        print("Method: :get_komo not implemented for Primtive: ", __name__)
        return [], [], []

    def is_grasping(self):
        # check if grasping is set in primitive, and if we are actually grasping
        if self.grasping and self.S.getGripperIsGrasping(self.gripper):
            self.C.attach(self.gripper, self.goal)
            return True
        # otherwise we are not grasping
        return False

    def is_open(self):
        # check if placing is set in primitive, and if gripper is open
        if self.placing and not self.S.getGripperIsGrasping(self.gripper) and \
                self.place_counter > self.max_place_counter:
            return True
        # otherwise we are still graping
        return False

    def is_done(self, t):
        if self.n_steps < t - self.t_start:
            return True
        return False

    def goal_changed_cond(self):
        current_goal_pos = self.C.frame(self.goal).getPosition()
        if np.isclose(current_goal_pos, self.initial_goal_position,
                      atol=0.04).all():
            return True
        else:
            return False

    def _get_overhead_for_q(self, q, overhead=constants.OVERHEAD_VIA):
        self.C.setJointState(q)
        gripper_pos_via = self.C.frame(self.gripper).getPosition()
        gripper_pos_via[2] += overhead
        iK = self.C.komo_IK(False)
        iK.addObjective(type=ry.OT.sos, feature=ry.FS.qItself, scale=[1e3] * 16, target=q)
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.position, frames=[self.gripper],
                        target=gripper_pos_via, scale=[1e1, 1e1, 1e2])
        iK.optimize()
        self.C.setFrameState(iK.getConfiguration(0))
        q_overhead = self.C.getJointState()
        return q_overhead

    def check_and_remove_start_overhead(self, phases, q_points, bezier_profiles, new_bezier):
        if not self.needs_overhead_start:
            phase_1 = phases.pop(0)
            phases[0] = round(phases[0] + phase_1, 1)
            del q_points[1]
            del bezier_profiles[0]
            bezier_profiles[0] = new_bezier


    def step(self, t):
        """
        Make a step of the primitive
        :param t: current time
        :return:
        """
        i = t - self.t_start
        if i < self.n_steps:
            if self.use_komo:
                self.C.setFrameState(self.komo.getConfiguration(i))
                q = self.C.getJointState()
            else:
                q = self.q_values[i]
                self.C.setJointState(q)
            self.S.step(q, self.tau, ry.ControlMode.position)
        elif self.grasping:
            self.S.closeGripper(self.gripper, speed=1.0)
            # check if gripper is graping
            self.C.setJointState(self.S.get_q())
            self.S.step([], self.tau, ry.ControlMode.none)
        elif self.placing:
            self.S.openGripper(self.gripper, speed=1.0)
            # check if griper is open
            if not self.S.getGripperIsGrasping(self.gripper) and not self.is_in_world:
                self.C.attach("world", self.goal)
                self.is_in_world = True
            self.place_counter = self.place_counter + 1
            self.C.setJointState(self.S.get_q())
            # for _ in range(10):
            self.S.step([], self.tau, ry.ControlMode.none)
        else:
            print("this condition should really not happen, did you forget to define a transition?")
        if not t % 50:
            self.V.setConfiguration(self.C)


class GravComp(Primitive):
    """
    Special class for holding the current position, waiting for an event to happen
    """

    def __init__(self, C, S, V, tau, n_steps, vis=False):
        Primitive.__init__(self, "grav_comp", C, S, V, tau, n_steps,
                           grasping=False, holding=False, placing=False, vis=vis)

    def create_primitive(self, t_start, gripper, goal, move_to=None):
        print(self.q_start)
        """
        Dont need a komo if nothing is happening
        :return:
        """
        return

    def step(self, t):
        """
        Also do nothing here
        """
        self.S.step([], self.tau, ry.ControlMode.none)
        self.V.setConfiguration(self.C)
        return

    def is_done_cond(self, t):
        """
        This primitive waits until another condition has been
        """
        return False


class Reset(Primitive):
    def __init__(self, C, S, V, tau, n_steps, komo=False, vis=False):
        Primitive.__init__(self, __class__.__name__, C, S, V, tau, n_steps,
                           grasping=False, holding=False, placing=False, komo=komo, vis=vis)

    def _get_goal_config(self, move_to=None):
        iK = self.C.komo_IK(False)

        q_reset = np.array([0., -1.,  0., -2.,  0.,  2.,  0.,  0.,  0., -1.,  0., -2.,  0.,  2.,  0.,  0.])
        # no contact
        iK.addObjective(type=ry.OT.sos, feature=ry.FS.qItself, target=self.q_start, scale=1e1*self.mask_gripper)
        iK.addObjective(type=ry.OT.eq, feature=ry.qItself, target=q_reset, scale=[1e1]*len(q_reset))
        iK.addObjective(type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1e3])
        iK.optimize()

        return iK.getConfiguration(0)

    def _get_komo(self, move_to=None):
        komo = self.C.komo_path(1, self.n_steps, self.duration, True)
        komo.addObjective(time=[], type=ry.OT.sos, feature=ry.FS.qItself, scale=[1e2] * 16, order=2)
        komo.addObjective(time=[1.0], type=ry.OT.eq, feature=ry.FS.qItself, target=self.q_goal,
                          scale=[1e2] * 16)
        komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1e2])
        # seems to have no effect
        komo.optimize()
        return komo

    def _get_joint_interpolation(self, move_to=None):

        q_via1 = self._get_overhead_for_q(self.q_start, overhead=0.2)

        phases = [0.3, 0.7]
        bezier_profiles = ["EaseInSine", "EaseOutSine"]
        q_points = [self.q_start, q_via1, self.q_goal]
        return phases, bezier_profiles, q_points


class Drop(Primitive):
    def __init__(self, C, S, V, tau, n_steps, komo=False, vis=False):
        Primitive.__init__(self, __class__.__name__, C, S, V, tau, n_steps,
                           grasping=False, holding=False, placing=False, komo=komo, vis=vis)

    def _get_goal_config(self, move_to=None):
        iK = self.C.komo_IK(False)

        # q_reset = np.array([0., -1.,  0., -2.,  0.,  2.,  0.,  0.,  0., -1.,  0., -2.,  0.,  2.,  0.,  0.])
        # # no contact
        # iK.addObjective(type=ry.OT.eq, feature=ry.qItself, target=q_reset, scale=[1e1]*len(q_reset))
        # iK.addObjective(type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1e3])
        # iK.optimize()

        self.C.setJointState(self.q_start)
        gripper_start_pos = self.C.frame(self.gripper).getPosition()
        gripper_start_pos[1] += 0.1
        gripper_start_pos[2] += 0.1
        iK = self.C.komo_IK(False)
        iK.addObjective(type=ry.OT.sos, feature=ry.FS.qItself, target=self.q_start)
        iK.addObjective(type=ry.OT.sos, feature=ry.FS.position, frames=[self.gripper],
                        target=gripper_start_pos, scale=[1e1] * 3)
        iK.addObjective(type=ry.OT.sos, feature=ry.vectorZ, frames=[self.gripper],
                        target=[0, 0, 1], scale=[1e1])
        iK.addObjective(type=ry.OT.sos, feature=ry.FS.distance, frames=[self.goal, self.gripper], scale=[-1e1])
        iK.addObjective(type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1e3])
        iK.optimize()
        self.C.setFrameState(iK.getConfiguration(0))
        # q_via1 = self.C.getJointState()

        return iK.getConfiguration(0)

    def _get_komo(self, move_to=None):
        komo = self.C.komo_path(1, self.n_steps, self.duration, True)
        komo.addObjective(time=[], type=ry.OT.sos, feature=ry.FS.qItself, scale=[1e2] * 16, order=2)
        komo.addObjective(time=[0.0, 0.5], type=ry.OT.sos, feature=ry.FS.vectorZ, frames=[self.gripper],
                          target=[0, 0, -1])
        komo.addObjective(time=[1.0], type=ry.OT.eq, feature=ry.FS.qItself, target=self.q_goal,
                          scale=[1e2] * 16)
        komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1e2])
        # seems to have no effect
        komo.optimize()
        return komo

    def _get_joint_interpolation(self, move_to=None):

        phases = [1.0]
        bezier_profiles = ["EaseInSine"]
        q_points = [self.q_start, self.q_goal]
        return phases, bezier_profiles, q_points


class TopGrasp(Primitive):

    def __init__(self, C, S, V, tau, n_steps, komo=False, vis=False):

        Primitive.__init__(self, __class__.__name__, C, S, V, tau, n_steps,
                           grasping=True, holding=False, placing=False, komo=komo, vis=vis)

    def _get_goal_config(self, move_to=None):
        block_size = self.C.frame(self.goal).getSize()
        iK = self.C.komo_IK(False)
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.positionRel, frames=[self.goal, self.gripper],
                        target=[0.0, 0.0, -(0.04 + block_size[2] / 2)], scale=[2e2])
        # iK.addObjective(type=ry.OT.sos, feature=ry.FS.positionDiff, frames=[self.goal, self.gripper],
        #                scale=[2])
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductZZ, frames=[self.gripper, self.goal],
                        target=[1], scale=[1])
        if block_size[0] > block_size[1]:
            iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductXX,
                            frames=[self.gripper, self.goal], target=[0], scale=[1])
            iK.addObjective(type=ry.OT.eq, feature=ry.qItself,
                            target=[block_size[1] / 2] * 16, scale=self.mask_gripper)
        else:
            iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductXY,
                            frames=[self.gripper, self.goal], target=[0], scale=[1])
            iK.addObjective(type=ry.OT.eq, feature=ry.qItself,
                            target=[block_size[0] / 2] * 16, scale=self.mask_gripper)
        # no contact
        iK.addObjective(type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1e3])
        iK.optimize()
        return iK.getConfiguration(0)

    def _get_komo(self, move_to=None):
        # generate motion
        komo = self.C.komo_path(1, self.n_steps, self.duration, True)
        komo.addObjective(time=[1.], type=ry.OT.eq, feature=ry.FS.qItself, target=self.q_goal,
                          scale=[1e3] * 16)
        komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[5e1])
        if self.needs_overhead_start:
            komo.addObjective(time=[0.0, 0.2], type=ry.OT.eq, feature=ry.FS.position, frames=[self.gripper],
                              target=[0, 0, 0.8], scale=[1e3] * 3, order=2)
        komo.addObjective(time=[0.8, 1.], type=ry.OT.sos, feature=ry.FS.position, frames=[self.gripper],
                          target=[0, 0, -0.8], scale=[1e3] * 3, order=2)
        komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1e3])
        return komo

    def _get_joint_interpolation(self, move_to=None):

        # first via point
        q_via1 = self._get_overhead_for_q(self.q_start)
        # second via point
        q_via2 = self._get_overhead_for_q(self.q_goal)

        values = []

        if self.needs_overhead_start:
            phases = [0.2, 0.5, 0.3]
            q_points = [self.q_start, q_via1, q_via2, self.q_goal]
            bezier_profiles = ["EaseInSine", "Linear", "EaseOutSine"]

        else:
            phases = [0.7, 0.3]
            bezier_profiles = ["EaseInSine", "EaseOutSine"]
            q_points = [self.q_start, q_via2, self.q_goal]
        return phases, bezier_profiles, q_points


class TopPlace(Primitive):

    def __init__(self, C, S, V, tau, n_steps, komo=False, vis=False):
        Primitive.__init__(self, __class__.__name__, C, S, V, tau, n_steps,
                           grasping=False, holding=False, placing=True, komo=komo, vis=vis)

    def _get_goal_config(self, move_to=None):
        # get current joint state
        iK = self.C.komo_IK(False)
        block_size = self.C.frame(self.goal).getSize()
        tower_placment = move_to
        tower_placment[2] = tower_placment[2] + (block_size[2] / 4) # really should only be half, but 4 works better
        iK.addObjective(type=ry.OT.sos, feature=ry.FS.qItself, target=self.q_start, scale=self.mask_gripper)
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.vectorX, frames=[self.gripper], target=[0, 1, 0])
        # we assume the object is attached to the frame of the gripper, therefore we can simply just
        # tell the goal object should have a position
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.position,
                        frames=[self.goal], scale=[1e2] * 3, target=tower_placment)
        # z-axis of gripper should align in z-axis of world frame
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.vectorZ, frames=[self.gripper], target=[0, 0, 1])
        # no contact
        iK.addObjective(type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions)
        iK.optimize()
        return iK.getConfiguration(0)

    def _get_komo(self, move_to=None):
        komo = self.C.komo_path(1, self.n_steps, self.duration, True)
        komo.addObjective(time=[], type=ry.OT.sos, feature=ry.FS.qItself, scale=[1e1] * 16, order=2)
        komo.addObjective(time=[0.8, 1], type=ry.OT.sos, feature=ry.FS.qItself, scale=[1e3] * 16, order=2)
        komo.addObjective(time=[], type=ry.OT.sos, feature=ry.FS.qItself, scale=np.asarray(self.mask_gripper) * 10,
                          target=self.q_start)
        komo.addObjective(time=[1.], type=ry.OT.eq, feature=ry.FS.qItself,
                          target=self.q_goal, scale=[1e1] * 16)

        komo.addObjective(time=[0.0, 0.2], type=ry.OT.sos, feature=ry.FS.position, frames=[self.gripper],
                          target=[0, 0, 0.8], scale=[1e3] * 3, order=2)
        komo.addObjective(time=[0.9, 1.], type=ry.OT.sos, feature=ry.FS.position, frames=[self.gripper],
                          target=[0, 0, -0.8], scale=[1e3] * 3, order=2)
        komo.addObjective(time=[], type=ry.OT.eq, feature=ry.FS.vectorZ, frames=[self.gripper], target=[0, 0, 1])
        komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1e1])
        return komo

    def _get_joint_interpolation(self, move_to=None):

        lift_of = self.C.frame(self.gripper).getPosition()
        lift_of[2] = lift_of[2] + 0.2

        # first via point
        iK = self.C.komo_IK(False)
        overhead_get = self.C.frame(self.gripper).getPosition()
        print(overhead_get)
        overhead_get[2] = overhead_get[2] + 0.2
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.vectorZ, frames=[self.gripper], target=[0, 0, 1])
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.position, frames=[self.gripper],
                        target=overhead_get, scale=[2e2])
        iK.addObjective(type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1e2])
        iK.optimize()
        self.C.setFrameState(iK.getConfiguration(0))
        q_via1 = self.C.getJointState()
        q_via1[7], q_via1[-1] = self.q_start[7], self.q_start[-1]

        if self.vis:
            self.V.setConfiguration(self.C)
            print("Displaying Configuration for Via Point 1")
            time.sleep(5)
        self.C.setFrameState(self.goal_config)
        move_to[2] = move_to[2] + 0.2

        iK = self.C.komo_IK(False)
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.position, frames=[self.gripper],
                        target=move_to, scale=[2e2])
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.vectorZ, frames=[self.gripper], target=[0, 0, 1])
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductZZ, frames=[self.gripper, self.goal], target=[1],
                        scale=[1])
        # iK.addObjective(type=ry.OT.eq, feature=ry.FS.distance, frames=[self.goal, self.gripper])
        # no contact
        iK.addObjective(type=ry.OT.sos, feature=ry.FS.accumulatedCollisions, scale=[1e2])
        iK.optimize()

        self.C.setFrameState(iK.getConfiguration(0))
        q_via2 = self.C.getJointState()

        q_via2 = self._get_overhead_for_q(self.q_goal)

        #q_via2[7], q_via2[-1] = self.q_start[7], self.q_start[-1]
        if self.vis:
            self.V.setConfiguration(self.C)
            print("showing via point 2")
            time.sleep(5)
        self.C.setFrameState(self.start_config)
        self.V.setConfiguration(self.C)

        # motion through via points is created here
        phases = [0.2, 0.6, 0.2]
        bezier_profiles = ["EaseInSine", "Linear", "EaseOutSine"]
        q_points = [self.q_start, q_via1, q_via2, self.q_goal]

        return phases, bezier_profiles, q_points


class SideGrasp(Primitive):

    def __init__(self, C, S, V, tau, n_steps, vis=False):
        Primitive.__init__(self, "side_grasp", C, S, V, tau, n_steps,
                           grasping=True, holding=False, placing=False, vis=vis)

    def _get_goal_config(self, move_to=None):
        # generate self.goal configuration
        iK = self.C.komo_IK(False)
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.positionRel, frames=[self.goal, self.gripper],
                        target=[0.0, 0.0, -0.07])
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductXX, frames=[self.goal, self.gripper], target=[-1],
                        scale=[1e2])
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductYZ, frames=[self.gripper, self.goal], target=[-1])
        iK.addObjective(type=ry.OT.ineq, feature=ry.FS.distance, frames=[self.goal, self.gripper], scale=[1e1])
        # no contact
        iK.addObjective(type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions)
        iK.optimize()
        return iK.getConfiguration(0)

    def _get_komo(self, move_to=None):
        komo = self.C.komo_path(1, self.n_steps, self.duration, True)
        komo.addObjective(time=[0.8, 1.], type=ry.OT.eq, feature=ry.FS.qItself, scale=[1e3] * 16, order=2)
        komo.addObjective(time=[1.], type=ry.OT.eq, feature=ry.FS.qItself, target=self.q_goal, scale=[1] * 16)
        komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1])
        komo.addObjective(time=[0.8, 1.], type=ry.OT.eq, feature=ry.FS.scalarProductYZ,
                          frames=[self.goal, self.gripper],
                          target=[-1], scale=[1e3])
        komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.distance, frames=[self.goal, self.gripper],
                          scale=[1e3])
        komo.optimize()
        return komo


class LiftUp(Primitive):

    def __init__(self, C, S, V, tau, n_steps, vis=False):
        Primitive.__init__(self, "lift_up", C, S, V, tau, n_steps,
                           grasping=False, holding=True, placing=False, vis=vis)

    def _get_goal_config(self, move_to=None):
        iK = self.C.komo_IK(False)
        # get new position
        goal_position = self.C.frame(self.gripper).getPosition()
        goal_position[2] = goal_position[2] + 1.0
        q = self.C.getJointState()
        iK.addObjective(type=ry.OT.sos, feature=ry.FS.qItself, target=q, scale=self.mask_gripper)
        iK.addObjective(type=ry.OT.sos, feature=ry.FS.position,
                        frames=[self.gripper], scale=[1] * 3, target=goal_position)
        # no contact
        iK.addObjective(type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions)
        iK.optimize()

        return iK.getConfiguration(0)

    def _get_komo(self, move_to=None):
        komo = self.C.komo_path(1, self.n_steps, self.duration, True)
        komo.addObjective(time=[], type=ry.OT.sos, feature=ry.FS.qItself, scale=[1e1] * 16, order=2)
        komo.addObjective(time=[1.], type=ry.OT.eq, feature=ry.FS.qItself, target=self.q_goal, scale=[1e2] * 16)
        komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1e1])
        komo.optimize()
        return komo


class PullIn(Primitive):

    def __init__(self, C, S, V, tau, n_steps, komo=False, vis=False):
        Primitive.__init__(self, __class__.__name__, C, S, V, tau, n_steps, komo=komo,
                           grasping=False, holding=False, placing=False, vis=vis)

    def _get_goal_config(self, move_to=None):
        block_size = self.C.frame(self.goal).getSize()
        xy_diag = np.sqrt(block_size[0] ** 2 + block_size[1] ** 2) / 2
        buffer = 0.04

        iK = self.C.komo_IK(False)
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.positionRel, frames=[self.goal, self.gripper],
                        target=[0, -(xy_diag + buffer), -0.05], scale=[1e1, 1e1, 1e1])  # height not so important
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.vectorX, frames=[self.gripper], target=[1, 0, 0])
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.vectorZ, frames=[self.gripper], target=[0, 0, 1])
        iK.addObjective(type=ry.OT.ineq, feature=ry.FS.distance, frames=[self.goal, self.gripper],
                        scale=[1e0])
        iK.addObjective(type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions)
        iK.optimize()

        return iK.getConfiguration(0)

    def _get_komo(self, move_to=None):
        self.C.setJointState(self.q_goal)
        aligned_gripper_position = self.C.frame(self.gripper).getPosition()
        aligned_gripper_position[1] = aligned_gripper_position[1] - 0.5
        print(aligned_gripper_position)

        komo = self.C.komo_path(1, self.n_steps, self.duration, True)
        komo.addObjective(time=[], type=ry.OT.sos, feature=ry.FS.qItself, scale=[1e1] * 16, order=2)
        komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.jointLimits)
        komo.addObjective(time=[0.5], type=ry.OT.eq, feature=ry.FS.qItself, target=self.q_goal,
                          scale=[1e2] * 16)
        komo.addObjective(time=[1.0], type=ry.OT.eq, feature=ry.FS.position, frames=[self.gripper],
                          target=aligned_gripper_position, scale=[1e2] * 3)
        # komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1])
        komo.addObjective(time=[0.5, 1.0], type=ry.OT.eq, feature=ry.FS.position, frames=[self.gripper],
                          target=[0, -0.8, 0], scale=[1e1] * 3, order=2)
        komo.optimize()
        return komo

    def _get_joint_interpolation(self, move_to=None):
        iK = self.C.komo_IK(False)
        self.C.setJointState(self.q_goal)
        gripper_pos_after_push = self.C.frame(self.gripper).getPosition()
        gripper_pos_after_push[1] = gripper_pos_after_push[1] - 0.1  # TODO calculate actual distance
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.position, frames=[self.gripper],
                        target=gripper_pos_after_push, scale=[1e1] * 3)
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.vectorX, frames=[self.gripper], target=[1, 0, 0])
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.vectorZ, frames=[self.gripper], target=[0, 0, 1])
        iK.addObjective(type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions)
        iK.optimize()

        self.C.setFrameState(iK.getConfiguration(0))
        q_via1 = self.C.getJointState()
        if self.vis:
            self.V.setConfiguration(self.C)
            print("Displaying Configuration for Via Point 1")
            time.sleep(5)
        self.C.setFrameState(self.goal_config)

        # motion through via points is created here
        phases = [0.5, 0.5]
        bezier_profiles = ["EaseInOutSine", "EaseInOutSine"]
        q_points = [self.q_start, self.q_goal, q_via1]

        return phases, bezier_profiles, q_points


class PushToEdge(Primitive):

    def __init__(self, C, S, V, tau, n_steps, komo=False, vis=False):
        Primitive.__init__(self, __class__.__name__, C, S, V, tau, n_steps,
                           grasping=False, holding=False, placing=False, komo=komo, vis=vis)

        self.push_direction = 0

    def _get_goal_config(self, move_to=None):
        # find out, which edge we are pushing
        block_pos = self.C.frame(self.goal).getPosition()
        # check if push to right or left edge of table
        if np.abs(self.right_edge_x - block_pos[0]) < np.abs(self.left_edge_x - block_pos[0]):
            self.push_direction = 1  # right
            print("Pushing to right edge")
        else:
            self.push_direction = -1  # left
            print("Pushing to left edge")

        iK = self.C.komo_IK(False)
        # get new position
        block_size = self.C.frame(self.goal).getSize()
        xy_diag = np.sqrt(block_size[0] ** 2 + block_size[1] ** 2) / 2
        buffer = 0.05
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.positionRel, frames=[self.goal, self.gripper],
                        target=[0, -self.push_direction * (xy_diag + buffer), -0.05], scale=[1e1, 1e1, 1e0])
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.vectorZ, frames=[self.gripper],
                        target=[0, 0, 1])
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductXX, frames=[self.gripper, "world"],
                        target=[0])

        # no contact
        iK.addObjective(type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions)
        iK.optimize()

        return iK.getConfiguration(0)

    def _get_komo(self, move_to=None):
        komo = self.C.komo_path(1, self.n_steps, self.duration, True)
        komo.addObjective(time=[], type=ry.OT.sos, feature=ry.FS.qItself, scale=[1e0] * 16, order=2)
        # komo.addObjective(time=[0.0, 0.2], type=ry.OT.eq, feature=ry.FS.position, frames=[self.gripper],
        #                  target=[0, 0.0, 0.2], scale=[1e1] * 3, order=2)
        # komo.addObjective(time=[0.8, 1.0], type=ry.OT.eq, feature=ry.FS.position, frames=[self.gripper],
        #                  target=[0, 0.0, -0.2], scale=[1e1] * 3, order=2)
        komo.addObjective(time=[1.0], type=ry.OT.eq, feature=ry.FS.qItself, target=self.q_goal,
                          scale=[1e2] * 16)
        komo.addObjective(time=[], type=ry.OT.eq, feature=ry.FS.accumulatedCollisions, scale=[1e2])
        komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.distance, frames=[self.goal, self.gripper],
                          scale=[1e2])
        # komo.addObjective(time=[0.5, 1.0], type=ry.OT.eq, feature=ry.FS.position, frames=[self.gripper],
        #                  target=[0, 0.5, 0], scale=[1e1] * 3, order=2)
        komo.optimize()
        return komo

    def _get_joint_interpolation(self, move_to=None):

        # get overhead for the current position
        iK = self.C.komo_IK(False)
        self.C.setJointState(self.q_start)
        print("Quaternion is: ", self.C.frame(self.gripper).getQuaternion())

        # get the overhead for via points

        # 1st via point
        q_via1 = self._get_overhead_for_q(self.q_start)

        # 2nd via point
        q_via2 = self._get_overhead_for_q(self.q_goal)

        # 3rd via point (to push motion)
        # find the push distance
        block_position_x = self.C.frame(self.goal).getPosition()[0]
        push_distance = 1 - np.abs(block_position_x)
        self.C.setJointState(self.q_goal)
        gripper_pose_via_3 = self.C.frame(self.gripper).getPosition()
        gripper_pose_via_3[0] = gripper_pose_via_3[0] + (push_distance * self.push_direction)
        iK = self.C.komo_IK(False)
        iK.addObjective(type=ry.OT.sos, feature=ry.FS.qItself, scale=[1e2] * 16, target=self.q_goal)
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.position, frames=[self.gripper],
                        target=gripper_pose_via_3, scale=[1e1, 1e1, 1e2])
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.vectorZ, frames=[self.gripper], target=[0, 0, 1])
        iK.optimize()
        self.C.setFrameState(iK.getConfiguration(0))
        q_via3 = self.C.getJointState()

        phases = [0.1, 0.4, 0.1, 0.4]
        bezier_profiles = ["EaseInOutSine", "EaseInOutSine", "EaseInOutSine", "EaseInOutSine"]
        q_points = [self.q_start, q_via1, q_via2, self.q_goal, q_via3]

        self.check_and_remove_start_overhead(phases, q_points, bezier_profiles, new_bezier="EaseInOutSine")

        return phases, bezier_profiles, q_points


class EdgeGrasp(Primitive):

    def __init__(self, C, S, V, tau, n_steps, komo=False, vis=False):
        Primitive.__init__(self, __class__.__name__, C, S, V, tau, n_steps, komo=komo,
                           grasping=True, holding=False, placing=False, vis=vis)
        self.push_direction = 0

    def _get_goal_config(self, move_to=None):
        print(self.q_start)
        # find out, which edge we are pushing
        block_pos = self.C.frame(self.goal).getPosition()
        # check if push to right or left edge of table
        if np.abs(self.right_edge_x - block_pos[0]) < np.abs(self.left_edge_x - block_pos[0]):
            self.push_direction = 1  # right grasp
        else:
            self.push_direction = -1  # left grasp
        block_size = self.C.frame(self.goal).getSize()

        # TODO actually calculate which corner is over edge, and closest to robot for grasping
        #  -> independent of orientation
        corner = block_size[:3] / 2
        corner[0] = self.push_direction * (corner[0] - 0.05)
        corner[1] = -corner[1] - 0.05
        corner[2] = 0
        iK = self.C.komo_IK(False)
        # iK.addObjective(type=ry.OT.sos, feature=ry.FS.qItself, scale=[1e0] * 16, target=self.q_start)
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.vectorZ, frames=[self.gripper],
                        target=[0, -1, 0], scale=[1e2])
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductXZ, frames=["world", self.gripper],
                        target=[0], scale=[1e2])
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductZZ, frames=[self.gripper, "world"],
                        target=[0], scale=[1e2])
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductXX, frames=[self.gripper, "world"],
                        target=[0], scale=[1e2])
        iK.addObjective(type=ry.OT.sos, feature=ry.FS.positionRel, frames=[self.gripper, self.goal],
                        target=corner, scale=[1e1] * 3)
        iK.addObjective(type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1e2])
        iK.addObjective(type=ry.OT.eq, feature=ry.qItself,
                        target=[block_size[2] / 2] * 16, scale=self.mask_gripper)
        iK.optimize()

        return iK.getConfiguration(0)

    def _get_komo(self, move_to=None):
        komo = self.C.komo_path(1, 200, self.duration, True)
        komo.addObjective(time=[], type=ry.OT.sos, feature=ry.FS.qItself, scale=[1e1] * 16, order=2)
        komo.addObjective(time=[1.0], type=ry.OT.eq, feature=ry.FS.qItself, target=self.q_goal,
                          scale=[1e2] * 16)
        #komo.addObjective(time=[0.6, 1.0], type=ry.OT.sos, feature=ry.FS.position, frames=[self.gripper],
        #                  target=[0, 0.1, 0], scale=[1e2] * 3, order=2)
        komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1e2])
        komo.optimize()
        return komo

    def _get_joint_interpolation(self, move_to=None):
        q_via1 = self._get_overhead_for_q(self.q_start)

        self.C.setJointState(self.q_goal)
        gripper_pose_via_2 = self.C.frame(self.gripper).getPosition()
        gripper_pose_via_2[1] -= 0.2
        iK = self.C.komo_IK(False)
        iK.addObjective(type=ry.OT.sos, feature=ry.FS.qItself, scale=[1e2] * 16, target=self.q_goal)
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.position, frames=[self.gripper],
                        target=gripper_pose_via_2, scale=[1e1, 1e1, 1e2])
        iK.optimize()
        self.C.setFrameState(iK.getConfiguration(0))
        q_via2 = self.C.getJointState()
        phases = [0.1, 0.7, 0.2]
        bezier_profiles = ["EaseInOutSine", "EaseInSine", "EaseOutSine"]
        q_points = [self.q_start, q_via1, q_via2, self.q_goal]

        # check if overhead is really needed
        self.check_and_remove_start_overhead(phases, q_points, bezier_profiles, new_bezier="EaseInSine")

        return phases, bezier_profiles, q_points


class EdgePlace(Primitive):

    def __init__(self, C, S, V, tau, n_steps, komo=False, vis=False):
        Primitive.__init__(self, __class__.__name__, C, S, V, tau, n_steps, komo=komo,
                           grasping=False, holding=False, placing=True, vis=vis)

    def _get_goal_config(self, move_to=None):
        # get current joint state
        iK = self.C.komo_IK(False)
        block_size = self.C.frame(self.goal).getSize()
        tower_placment = move_to
        tower_placment[2] = tower_placment[2] + block_size[2] / 2
        iK.addObjective(type=ry.OT.sos, feature=ry.FS.qItself, target=self.q_start, scale=self.mask_gripper)
        # we assume the object is attached to the frame of the gripper, therefore we can simply just
        # tell the goal object should have a position
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.position,
                        frames=[self.goal], scale=[1e1] * 3, target=tower_placment)
        # z-axis of gripper should align in z-axis of world frame
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductXZ, frames=["world", self.goal],
                        target=[0], scale=[1e1])
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductYZ, frames=["world", self.goal],
                        target=[0], scale=[1e1])
        # no contact
        iK.addObjective(type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions)
        iK.optimize()
        return iK.getConfiguration(0)

    def _get_komo(self, move_to=None):
        komo = self.C.komo_path(1, self.n_steps, self.duration, True)
        komo.addObjective(time=[], type=ry.OT.sos, feature=ry.FS.qItself, scale=[1e2] * 16, order=2)
        komo.addObjective(time=[1.0], type=ry.OT.eq, feature=ry.FS.qItself, target=self.q_goal,
                          scale=[1e2] * 16)
        komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1e2])
        # seems to have no effect
        # komo.addObjective(time=[0.9, 1.], type=ry.OT.sos, feature=ry.FS.position, frames=[self.gripper],
        #                   target=[0, 0, -0.2], scale=[1e1] * 3, order=2)
        komo.optimize()
        return komo

    def _get_joint_interpolation(self, move_to=None):
        self.C.setJointState(self.q_goal)
        gripper_pose_via_1 = self.C.frame(self.gripper).getPosition()
        gripper_pose_via_1[1] -= 0.3
        iK = self.C.komo_IK(False)
        # iK.addObjective(type=ry.OT.eq, feature=ry.FS.position, frames=[self.gripper],
        #                target=gripper_pose_via_2, scale=[1e1] * 3)
        block_size = self.C.frame(self.goal).getSize()
        xy_diag = np.sqrt(block_size[0] ** 2 + block_size[1] ** 2) / 2
        buffer = 0.01
        iK.addObjective(type=ry.OT.sos, feature=ry.FS.qItself, scale=[1e2] * 16, target=self.q_goal)
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.position, frames=[self.gripper],
                        target=gripper_pose_via_1, scale=[1e1, 1e1, 1e2])
        iK.optimize()
        self.C.setFrameState(iK.getConfiguration(0))
        q_via1 = self.C.getJointState()

        if self.needs_overhead_start:
            self.C.setJointState(self.q_start)
            gripper_pose_via_1 = self.C.frame(self.gripper).getPosition()
            gripper_pose_via_1[2] += 0.3
            iK = self.C.komo_IK(False)
            iK.addObjective(type=ry.OT.sos, feature=ry.FS.qItself, scale=[1e2] * 16, target=self.q_start)
            iK.addObjective(type=ry.OT.eq, feature=ry.FS.position, frames=[self.gripper],
                            target=gripper_pose_via_1, scale=[1e1, 1e1, 1e2])
            iK.optimize()
            self.C.setFrameState(iK.getConfiguration(0))
            q_via11 = self.C.getJointState()
            phases = [0.3, 0.3, 0.4]
            bezier_profiles = ["EaseInOutSine", "EaseInSine", "EaseOutSine"]
            q_points = [self.q_start, q_via11, q_via1, self.q_goal]
            return phases, bezier_profiles, q_points
        # motion through via points is created here
        phases = [0.6, 0.4]
        bezier_profiles = ["EaseInSine", "EaseOutSine"]
        q_points = [self.q_start, q_via1, self.q_goal]

        return phases, bezier_profiles, q_points


class AngleEdgePlace(Primitive):

    def __init__(self, C, S, V, tau, n_steps, komo=False, vis=False):
        Primitive.__init__(self, __class__.__name__, C, S, V, tau, n_steps, komo=komo,
                           grasping=False, holding=False, placing=True, vis=vis)

    def _get_goal_config(self, move_to=None):
        # get current joint state
        iK = self.C.komo_IK(False)
        block_size = self.C.frame(self.goal).getSize()
        tower_placement = move_to
        tower_placement[2] += 0.1 + block_size[2] / 2
        # iK.addObjective(type=ry.OT.sos, feature=ry.FS.qItself, target=self.q_start, scale=self.mask_gripper)
        # we assume the object is attached to the frame of the gripper, therefore we can simply just
        # tell the goal object should have a position
        iK.addObjective(type=ry.OT.sos, feature=ry.FS.qItself, target=self.q_start, scale=[1e1] * 16)
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.position,
                        frames=[self.goal], scale=[1e2] * 3, target=tower_placement)
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.vectorZ,
                        frames=[self.goal], scale=[1e1] * 3, target=[0, -0.5, 1])
        # iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductXZ, frames=["world", self.goal],
        #                target=[0], scale=[1e1])
        # iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductYZ, frames=["world", self.goal],
        #                 target=[0], scale=[1e1])
        # no contact
        iK.addObjective(type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1e2])
        iK.optimize()
        return iK.getConfiguration(0)

    def _get_komo(self, move_to=None):
        komo = self.C.komo_path(1, self.n_steps, self.duration, True)
        komo.addObjective(time=[], type=ry.OT.sos, feature=ry.FS.qItself, scale=[1e2] * 16, order=2)
        komo.addObjective(time=[1.0], type=ry.OT.eq, feature=ry.FS.qItself, target=self.q_goal,
                          scale=[1e2] * 16)
        komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1e2])
        # seems to have no effect
        komo.optimize()
        return komo

    def _get_joint_interpolation(self, move_to=None):
        overhead = 0.2

        # 1st via point
        q_via1 = self._get_overhead_for_q(self.q_start, 0.5)

        # 2nd via point
        q_via2 = self._get_overhead_for_q(self.q_goal)

        phases = [0.2, 0.6, 0.2]
        bezier_profiles = ["EaseInOutSine", "EaseInOutSine", "EaseInOutSine"]
        q_points = [self.q_start, q_via1, q_via2, self.q_goal]

        return phases, bezier_profiles, q_points
