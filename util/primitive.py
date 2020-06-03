import libry as ry
import numpy as np
import time
from transitions import State
from functools import partial

from scipy.io.matlab.mio5_params import mat_struct


class Primitive(State):

    def __init__(self, name, C, S, V, tau, n_steps, gripper, goal="goal",
                 grasping=False, holding=False, placing=False, vis=False):

        State.__init__(self, name, on_enter="init_state")
        self.gripper = gripper
        self.goal = goal
        self.duration = tau * n_steps
        self.n_steps = n_steps
        self.tau = tau
        self.C = C
        self.S = S
        self.V = V
        self.grasping = grasping
        self.holding = holding
        self.placing = placing
        self.initial_goal_position = self.C.frame(self.goal).getPosition()
        self.vis = vis

        # mask to make sure the fingers do not change
        self.mask_gripper = [0] * 16
        self.mask_gripper[-1] = 1
        self.mask_gripper[7] = 1

        # t_start and komo need initialized in the init() function
        self.t_start = None
        self.start_config = None
        self.goal_config = None
        self.goal_joint_config = None
        self.iK = None
        self.komo = None

    def create_primitive(self, t_start, move_to=None):
        self.t_start = t_start
        self.start_config = self.C.getFrameState()
        # get the goal configuration
        self.goal_config = self._get_goal_config(move_to)
        self.C.setFrameState(self.goal_config)

        # visualize goal config if V is set
        if self.vis:
            print(f"Displaying Goal Config of Primitive: {self.name}")
            self.V.setConfiguration(self.C)
            time.sleep(5)
        # reset initial config in configuration space
        self.goal_joint_config = self.C.getJointState()
        self.C.setFrameState(self.start_config)

        # get the komo path for the primitive and optimize
        self.komo = self._get_komo(move_to)
        self.komo.optimize(False)
        # visualize komo path if V is set
        if self.vis:
            V2 = self.komo.view()
            time.sleep(2)
            V2.playVideo()
            time.sleep(2)

    def _get_goal_config(self, move_to=None):
        print("Method: :get_goal_config not implemented for Primtive: ", __name__)
        return

    def _get_komo(self, move_to=None):
        print("Method: :get_komo not implemented for Primtive: ", __name__)
        return

    def is_grasping(self):
        # check if grasping is set in primitive, and if we are actually grasping
        if self.grasping and self.S.getGripperIsGrasping(self.gripper):
            self.C.attach(self.gripper, self.goal)
            return True
        # otherwise we are not grasping
        return False

    def is_open(self):
        # check if placing is set in primitive, and if gripper is open
        if self.placing and not self.S.getGripperIsGrasping(self.gripper):
            self.C.attach("world", self.goal)
            return True
        # otherwise we are still graping
        return False

    def goal_changed_cond(self):
        current_goal_pos = self.C.frame(self.goal).getPosition()
        if np.isclose(current_goal_pos, self.initial_goal_position,
                      atol=0.04).all():
            return True
        else:
            return False

    def step(self, t, goal_current=None):
        i = t - self.t_start
        if i < self.n_steps:
            self.C.setFrameState(self.komo.getConfiguration(i))
            q = self.C.getJointState()
            self.S.step(q, self.tau, ry.ControlMode.position)
        elif self.grasping:
            self.S.closeGripper(self.gripper, speed=1.0)
            # check if gripper is graping
            if self.S.getGripperIsGrasping(self.gripper):
                self.C.attach(self.gripper, self.goal)
            self.C.setJointState(self.S.get_q())
            self.S.step([], self.tau, ry.ControlMode.none)
        elif self.placing:
            self.S.openGripper(self.gripper, speed=1.0)
            # check if griper is open
            if not self.S.getGripperIsGrasping(self.gripper):
                self.C.attach("world", self.goal)
            self.C.setJointState(self.S.get_q())
            self.S.step([], self.tau, ry.ControlMode.none)
        else:
            print("this condition should really not happen, did you forget to define a transition?")
        self.V.setConfiguration(self.C)


class GravComp(Primitive):
    """
    Special class for holding the current position, waiting for an event to happen
    """

    def __init__(self, C, S, V, tau, n_steps, gripper, goal="goal", vis=False):
        Primitive.__init__(self, "grav_comp", C, S, V, tau, n_steps, gripper, goal,
                           grasping=False, holding=False, placing=False, vis=vis)

    def create_primitive(self, t_start, move_to=None):
        """
        Dont need a komo if nothing is happening
        :return:
        """
        return

    def step(self, t, goal_current=None):
        """
        Also do nothing here
        """
        self.S.step([], self.tau, ry.ControlMode.none)
        return

    def is_done_cond(self, t):
        """
        This primitive waits until another condition has been
        """
        return False


class TopGrasp(Primitive):

    def __init__(self, C, S, V,tau, n_steps, gripper, goal="goal", vis=False):
        Primitive.__init__(self, "top_grasp", C, S, V, tau, n_steps, gripper, goal,
                           grasping=True, holding=False, placing=False, vis=vis)
        
    def _get_goal_config(self, move_to=None):
        iK = self.C.komo_IK(False)
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.positionRel, frames=[self.goal, self.gripper], target=[0.0, 0.0, -0.07],
                        scale=[1])
        iK.addObjective(type=ry.OT.sos, feature=ry.FS.positionDiff, frames=[self.goal, self.gripper], target=[0.0, 0.0, 0.0],
                        scale=[2])
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductZZ, frames=[self.gripper, self.goal], target=[1], scale=[1])
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductXY, frames=[self.gripper, self.goal], target=[1], scale=[1])
        iK.addObjective(type=ry.OT.ineq, feature=ry.FS.distance, frames=[self.goal, self.gripper])
        # no contact
        iK.addObjective(type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions)
        iK.optimize()

        return iK.getConfiguration(0)

    def _get_komo(self, move_to=None):
        # generate motion
        komo = self.C.komo_path(1, self.n_steps, self.duration, True)
        komo.addObjective(time=[0.8, 1.], type=ry.OT.eq, feature=ry.FS.qItself, scale=[1e3] * 16, order=2)
        komo.addObjective(time=[1.], type=ry.OT.eq, feature=ry.FS.qItself, target=self.goal_joint_config,
                          scale=[10] * 16)
        komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1])
        # komo.addObjective(time=[0.9, 1.], type=ry.OT.eq, feature=ry.FS.scalarProductZZ, frames=[self.gripper, self.goal],
        #                 target=[1], scale=[1])
        # komo.addObjective(time=[0.9, 1.], type=ry.OT.eq, feature=ry.FS.scalarProductXY, frames=[self.gripper, self.goal],
        #                 target=[1], scale=[1])
        komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.distance, frames=[self.goal, self.gripper], scale=[1])
        return komo


class TopPlace(Primitive):

    def __init__(self, C, S, V, tau, n_steps, gripper="R_gripper", goal="goal", vis=False):
        Primitive.__init__(self, __name__, C, S, V, tau, n_steps, gripper, goal,
                           grasping=False, holding=False, placing=True, vis=vis)

    def _get_goal_config(self, move_to=None):
        # get gripper and goal object positions
        #gripper_pos = np.asarray(self.C.frame(self.gripper).info()["X"])[:3]
        #goal_pos = np.asarray(self.C.frame(self.goal).info()["X"])[:3]
        # we only care about difference in z-axis
        #dist_to_ground = (gripper_pos - goal_pos)[2]
        # adjust where we need to move the gripper
        #move_to[2] = move_to[2] + dist_to_ground
        # get current joint state
        q = self.C.getJointState()
        iK = self.C.komo_IK(False)
        iK.addObjective(type=ry.OT.sos, feature=ry.FS.qItself, target=q, scale=self.mask_gripper)
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.position,
                        frames=[self.goal], scale=[1] * 3, target=move_to)
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.vectorZ, frames=[self.gripper], target=[0, 0, 1])
        # no contact
        iK.addObjective(type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions)
        iK.optimize()
        return iK.getConfiguration(0)

    def _get_komo(self, move_to=None):
        komo = self.C.komo_path(1, self.n_steps, self.duration, True)
        komo.addObjective(time=[], type=ry.OT.sos, feature=ry.FS.qItself, scale=[1e1] * 16, order=2)
        komo.addObjective(time=[1.], type=ry.OT.eq, feature=ry.FS.qItself,
                          target=self.goal_joint_config, scale=[1e2] * 16)
        komo.addObjective(time=[], type=ry.OT.eq, feature=ry.FS.vectorZ, frames=[self.gripper], target=[0, 0, 1])
        komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1e1])
        return komo


class SideGrasp(Primitive):

    def __init__(self, C, S, V, tau, n_steps, gripper="R_gripper", goal="goal", vis=False):
        Primitive.__init__(self, "side_grasp", C, S, V, tau, n_steps, gripper, goal,
                           grasping=True, holding=False, placing=False, vis=vis)

    def create_primitive(self, t_start, move_to=None):
        self.t_start = t_start
        start_config = self.C.getFrameState()

        # generate self.goal configuration
        iK = self.C.komo_IK(False)
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.positionRel, frames=[self.goal, self.gripper], target=[0.0, 0.0, -0.07])
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductXX, frames=[self.goal, self.gripper], target=[-1], scale=[1e2])
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductYZ, frames=[self.gripper, self.goal], target=[-1])
        iK.addObjective(type=ry.OT.ineq, feature=ry.FS.distance, frames=[self.goal, self.gripper], scale=[1e1])
        # no contact
        iK.addObjective(type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions)
        iK.optimize()
        self.goal_config = iK.getConfiguration(0)

        # set config to get joint config
        self.C.setFrameState(self.goal_config)
        if self.V:
            print("Displaying Side Grasp Config")
            self.V.setConfiguration(self.C)
            time.sleep(2)
        self.goal_joint_config = self.C.getJointState()
        self.C.setFrameState(start_config)

        # generate motion
        komo = self.C.komo_path(1, self.n_steps, self.duration, True)
        komo.addObjective(time=[0.8, 1.], type=ry.OT.eq, feature=ry.FS.qItself, scale=[1e3] * 16, order=2)
        komo.addObjective(time=[1.], type=ry.OT.eq, feature=ry.FS.qItself, target=self.goal_joint_config, scale=[1] * 16)
        komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1])
        komo.addObjective(time=[0.8, 1.], type=ry.OT.eq, feature=ry.FS.scalarProductYZ, frames=[self.goal, self.gripper],
                          target=[-1], scale=[1e3])
        komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.distance, frames=[self.goal, self.gripper], scale=[1e3])
        komo.optimize()
        self.komo = komo
        return


class LiftUp(Primitive):
    
    def __init__(self, C, S, V, tau, n_steps, gripper, goal="goal", vis=False):
        Primitive.__init__(self,"lift_up", C, S, V, tau, n_steps, gripper, goal,
                           grasping=False, holding=True, placing=False, vis=vis)
        
    def create_primitive(self, t_start, move_to=None):
        self.t_start = t_start
        start_config = self.C.getFrameState()
        iK = self.C.komo_IK(False)

        # get new position
        goal_position = self.C.frame(self.gripper).getPosition()
        goal_position[2] = goal_position[2] + 1.0
        q = self.C.getJointState()
        # print(q)
        mask_gripper = [0] * 16
        # mask_self.gripper[-1] = 1
        # mask_self.gripper[7] = 1
        iK.addObjective(type=ry.OT.sos, feature=ry.FS.qItself, target=q, scale=mask_gripper)
        iK.addObjective(type=ry.OT.sos, feature=ry.FS.position,
                        frames=[self.gripper], scale=[1] * 3, target=goal_position)
        # no contact
        iK.addObjective(type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions)
        iK.optimize()

        goal_config = iK.getConfiguration(0)
        # set config to get joint config
        self.C.setFrameState(goal_config)
        if self.V:
            print("Displaying Lift-Up   Config")
            self.V.setConfiguration(self.C)
            time.sleep(2)
        goal_joint_config = self.C.getJointState()
        self.C.setFrameState(start_config)
        komo = self.C.komo_path(1, self.n_steps, self.duration, True)
        komo.addObjective(time=[], type=ry.OT.sos, feature=ry.FS.qItself, scale=[1e1] * 16, order=2)
        komo.addObjective(time=[1.], type=ry.OT.eq, feature=ry.FS.qItself, target=goal_joint_config, scale=[1e2] * 16)
        komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1e1])
        komo.optimize()
        self.komo = komo
        return komo

