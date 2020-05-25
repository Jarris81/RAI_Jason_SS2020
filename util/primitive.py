import libry as ry
import time

from scipy.io.matlab.mio5_params import mat_struct


class Primitive:

    def __init__(self, C, S, primitive, tau, start, n_steps, gripper, goal="goal",
                 V=None, grasping=False, hold=False, releasing=False):

        self.gripper = gripper
        self.goal = goal
        self.start = start
        self.n_steps = n_steps
        self.tau = tau
        self.C = C
        self.S = S
        self.grasping = grasping

        # generate komo with specific primitive
        self.komo = primitive(C, n_steps, n_steps*tau, goal, gripper, V, hold)

    def is_done(self, t):
        i = t - self.start
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

    def step(self, t):
        i = t - self.start
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


def lift_up(C, n_steps, duration, gripper, goal, V, hold=False):
    start_config = C.getFrameState()
    iK = C.komo_IK(False)

    # get new position
    goal_position = C.frame(gripper).getPosition()
    goal_position[2] = goal_position[2] + 1.5
    q = C.getJointState()
    print(q)
    mask_gripper = [0] * 16
    mask_gripper[-1] = 1
    mask_gripper[7] = 1
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
        print("Displaying Lift Up Config")
        V.setConfiguration(C)
        time.sleep(5)
    goal_joint_config = C.getJointState()
    C.setFrameState(start_config)
    komo = C.komo_path(1, n_steps, duration, True)
    komo.addObjective(time=[], type=ry.OT.sos, feature=ry.FS.qItself, scale=[1e1] * 16, order=2)
    komo.addObjective(time=[1.], type=ry.OT.eq, feature=ry.FS.qItself, target=goal_joint_config, scale=[1e2] * 16)
    komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1e1])
    komo.optimize()
    return komo


def top_grasp(C, n_steps, duration, gripper, goal, V, hold=False):
    start_config = C.getFrameState()
    iK = C.komo_IK(False)
    # mask for gripper
    iK.addObjective(type=ry.OT.eq, feature=ry.FS.positionRel, frames=[gripper, goal], target=[0.0, 0.0, -0.07], scale=[1e2]*3)
    iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductZZ, frames=[goal, gripper], target=[1], scale=[3])
    iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductXY, frames=[goal, gripper], target=[1], scale=[3])
    iK.addObjective(type=ry.OT.ineq, feature=ry.FS.distance, frames=[gripper, goal])
    # no contact
    iK.addObjective(type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions)
    iK.optimize()

    goal_config = iK.getConfiguration(0)

    # set config to get joint config
    C.setFrameState(goal_config)
    if V:
        print("Displaying Top Grasp Config")
        V.setConfiguration(C)
        time.sleep(5)
    goal_joint_config = C.getJointState()
    C.setFrameState(start_config)

    # generate motion
    komo = C.komo_path(1, n_steps, duration, True)
    komo.addObjective(time=[], type=ry.OT.sos, feature=ry.FS.qItself, scale=[1e2] * 16, order=2)
    komo.addObjective(time=[1.], type=ry.OT.eq, feature=ry.FS.qItself, target=goal_joint_config, scale=[1e1] * 16)
    komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1e1])
    komo.addObjective(time=[0.8, 1.], type=ry.OT.eq, feature=ry.FS.scalarProductZZ, frames=[goal, gripper], target=[1], scale=[1e3])
    komo.addObjective(time=[0.8, 1.], type=ry.OT.eq, feature=ry.FS.scalarProductXY, frames=[goal, gripper], target=[1], scale=[1e3])
    komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.distance, frames=[gripper, goal], scale=[1e3])
    komo.optimize()
    return komo


def side_grasp(C, n_steps, duration, gripper, goal, V, hold=False):
    start_config = C.getFrameState()

    # generate goal configuration
    iK = C.komo_IK(False)
    iK.addObjective(type=ry.OT.eq, feature=ry.FS.positionRel, frames=[gripper, goal], target=[0.0, 0.0, -0.02])
    iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductXX, frames=[gripper, goal], target=[-1])
    iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductYZ, frames=[goal, gripper], target=[-1])
    iK.addObjective(type=ry.OT.ineq, feature=ry.FS.distance, frames=[gripper, goal])
    # no contact
    iK.addObjective(type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions)
    iK.optimize()
    goal_config = iK.getConfiguration(0)

    # set config to get joint config
    C.setFrameState(goal_config)
    if V:
        print("Displaying Side Grasp Config")
        V.setConfiguration(C)
        time.sleep(5)
    goal_joint_config = C.getJointState()
    C.setFrameState(start_config)

    # generate motion
    komo = C.komo_path(1, n_steps, duration, True)
    komo.addObjective(time=[], type=ry.OT.sos, feature=ry.FS.qItself, scale=[1e1] * 16, order=2)
    komo.addObjective(time=[1.], type=ry.OT.eq, feature=ry.FS.qItself, target=goal_joint_config, scale=[1e2] * 16)
    komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1e1])
    komo.addObjective(time=[0.8, 1.], type=ry.OT.eq, feature=ry.FS.scalarProductYZ, frames=[gripper, goal], target=[-1], scale=[3])
    komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.distance, frames=[gripper, goal], scale=[1e2])
    komo.optimize()
    return komo


def get_motion(C, primitive, n_steps, duration, gripper, goal, V, hold=False):

    start_config = C.getFrameState()
    goal_config = primitive(C, gripper, goal)
    C.setFrameState(goal_config)
    V.setConfiguration(C)
    time.sleep(2)
    goal_joint_config = C.getJointState()
    C.setFrameState(start_config)


