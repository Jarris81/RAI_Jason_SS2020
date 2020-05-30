import libry as ry
import numpy as np
import time
from transitions import State
from functools import partial

from scipy.io.matlab.mio5_params import mat_struct


class Primitive(State):

    def __init__(self, name, C, S, tau, n_steps, gripper, goal="goal",
                 V=None, grasping=False, holding=False, releasing=False):

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
        self.releasing = releasing
        self.initial_goal_position = self.C.frame(self.goal).getPosition()

        # t_start and komo need initialized in the init() function
        self.t_start = None
        self.komo = None

    def create_komo(self, t):
        print(" method is not implemented!!")
        return None

    def is_grasping(self):
        return self.S.getGripperIsGrasping(self.gripper)

    def is_done_cond(self, t):
        i = t - self.t_start
        if self.grasping:
            if i < self.n_steps or not self.S.getGripperIsGrasping(self.gripper):
                return False
            else:
                if self.S.getGripperIsGrasping(self.gripper):
                    self.C.attach(self.gripper, self.goal)
                    self.C.frame(self.goal).setContact(0)
                return True
        else:
            if i < self.n_steps:
                return False
            else:
                return True

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
            self.S.closeGripper(self.gripper, speed=5.0)
            q = self.S.get_q()
            q[-1] = self.S.getGripperWidth(self.gripper)
            self.C.setJointState(q)
            self.S.step([], self.tau, ry.ControlMode.none)


class GravComp(Primitive):
    """
    Special class for holding the current position, waiting for an event to happen
    """

    def __init__(self, C, S, tau, n_steps, gripper, goal="goal", V=None):
        Primitive.__init__(self, "grav_comp", C, S, tau, n_steps, gripper, goal, V,
                           grasping=False, holding=False, releasing=False)


    def create_komo(self, t):
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
        :param t:
        :return:
        """
        False


class TopGrasp(Primitive):

    def __init__(self, C, S, tau, n_steps, gripper, goal="goal", V=None):
        Primitive.__init__(self, "top_grasp", C, S, tau, n_steps, gripper, goal, V,
                           grasping=True, holding=False, releasing=False)
        
    def create_komo(self, t_start):
        self.t_start = t_start
        start_config = self.C.getFrameState()
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

        self.goal_config = iK.getConfiguration(0)

        # set config to get joint config
        self.C.setFrameState(self.goal_config)
        if self.V:
            print("Displaying Top Grasp Config")
            self.V.setConfiguration(self.C)
            time.sleep(2)

        self.goal_joint_config = self.C.getJointState()
        self.C.setFrameState(start_config)

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
        komo.optimize()

        V2 = komo.view()
        time.sleep(2)
        V2.playVideo()
        time.sleep(2)
        self.komo = komo
        return komo


class SideGrasp(Primitive):

    def __init__(self, C, S, tau, n_steps, gripper="R_gripper", goal="goal", V=None):
        Primitive.__init__(self,"side_grasp", C, S, tau, n_steps, gripper, goal, V,
                           grasping=True, holding=False, releasing=False)

    def create_komo(self, t_start):
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
    
    def __init__(self, C, S, tau, n_steps, gripper, goal="goal", V=None):
        Primitive.__init__(self,"lift_up", C, S, tau, n_steps, gripper, goal, V,
                           grasping=False, holding=True, releasing=False)
        
    def create_komo(self, t_start):
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
        

def lift_up(C, n_steps, duration, gripper, goal, V, hold=False):
    start_config = C.getFrameState()
    iK = C.komo_IK(False)

    # get new position
    goal_position = C.frame(gripper).getPosition()
    goal_position[2] = goal_position[2] + 1.0
    q = C.getJointState()
    # print(q)
    mask_gripper = [0] * 16
    # mask_gripper[-1] = 1
    # mask_gripper[7] = 1
    iK.addObjective(type=ry.OT.sos, feature=ry.FS.qItself, target=q, scale=mask_gripper)
    iK.addObjective(type=ry.OT.sos, feature=ry.FS.position,
                    frames=[gripper], scale=[1] * 3, target=goal_position)
    # no contact
    iK.addObjective(type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions)
    iK.optimize()

    goal_config = iK.getConfiguration(0)
    # set config to get joint config
    C.setFrameState(goal_config)
    if V:
        print("Displaying Lift-Up   Config")
        V.setConfiguration(C)
        time.sleep(2)
    goal_joint_config = C.getJointState()
    C.setFrameState(start_config)
    komo = C.komo_path(1, n_steps, duration, True)
    komo.addObjective(time=[], type=ry.OT.sos, feature=ry.FS.qItself, scale=[1e1] * 16, order=2)
    komo.addObjective(time=[1.], type=ry.OT.eq, feature=ry.FS.qItself, target=goal_joint_config, scale=[1e2] * 16)
    komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1e1])
    komo.optimize()
    return komo





def side_grasp(C, n_steps, duration, gripper, goal, V, hold=False):
    start_config = C.getFrameState()

    # generate goal configuration
    iK = C.komo_IK(False)
    iK.addObjective(type=ry.OT.eq, feature=ry.FS.positionRel, frames=[gripper, goal], target=[0.0, 0.0, -0.07])
    iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductXX, frames=[gripper, goal], target=[-1], scale=[1e2])
    iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductYZ, frames=[goal, gripper], target=[-1])
    iK.addObjective(type=ry.OT.ineq, feature=ry.FS.distance, frames=[gripper, goal], scale=[1e1])
    # no contact
    iK.addObjective(type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions)
    iK.optimize()
    goal_config = iK.getConfiguration(0)

    # set config to get joint config
    C.setFrameState(goal_config)
    if V:
        print("Displaying Side Grasp Config")
        V.setConfiguration(C)
        time.sleep(2)
    goal_joint_config = C.getJointState()
    C.setFrameState(start_config)

    # generate motion
    komo = C.komo_path(1, n_steps, duration, True)
    komo.addObjective(time=[0.8,1.], type=ry.OT.eq, feature=ry.FS.qItself, scale=[1e3] * 16, order=2)
    komo.addObjective(time=[1.], type=ry.OT.eq, feature=ry.FS.qItself, target=goal_joint_config, scale=[1] * 16)
    komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1])
    komo.addObjective(time=[0.8, 1.], type=ry.OT.eq, feature=ry.FS.scalarProductYZ, frames=[gripper, goal],
                      target=[-1], scale=[1e3])
    komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.distance, frames=[gripper, goal], scale=[1e3])
    komo.optimize()
    self.komo = komo
    return komo
