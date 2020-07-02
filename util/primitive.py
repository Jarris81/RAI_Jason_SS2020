import libry as ry
import numpy as np
import time
from transitions import State
import util.bezier as beziers
from functools import partial




from scipy.io.matlab.mio5_params import mat_struct


class Primitive(State):

    def __init__(self, name, C, S, V, tau, n_steps,
                 grasping=False, holding=False, placing=False,
                 interpolation=False, vis=False):

        State.__init__(self, name)
        self.duration = tau * n_steps
        self.n_steps = n_steps
        self.tau = tau
        self.C = C
        self.S = S
        self.V = V
        self.grasping = grasping
        self.holding = holding
        self.placing = placing
        self.vis = vis
        self.use_interpolation = interpolation
        self.max_place_counter = 100

        # mask to make sure the fingers do not change
        self.mask_gripper = [0] * 16
        self.mask_gripper[-1] = 1
        self.mask_gripper[7] = 1

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

    def create_primitive(self, t_start, gripper, goal, move_to=None):
        print("creating new primitive!")
        self.t_start = t_start
        self.gripper = gripper
        self.goal = goal
        self.q_start = self.C.getJointState()
        self.start_config = self.C.getFrameState()
        self.initial_goal_position = self.C.frame(self.goal).getPosition()
        # get the goal configuration
        self.goal_config = self._get_goal_config(move_to)
        self.C.setFrameState(self.goal_config)

        self.place_counter = 0
        self.is_in_world =False

        # visualize goal config if V is set
        if self.vis:
            print(f"Displaying Goal Config of Primitive: {self.name}")
            self.V.setConfiguration(self.C)
            time.sleep(5)
        # reset initial config in configuration space
        self.q_goal = self.C.getJointState()
        self.C.setFrameState(self.start_config)
        if self.use_interpolation:
            self.q_values = self._get_joint_interpolation(move_to=move_to)
        else:
            # get the komo path for the primitive and optimize
            self.komo = self._get_komo(move_to)
            self.komo.optimize(False)
            # visualize komo path if V is set
            if self.vis:
                V2 = self.komo.view()
                time.sleep(2)
                V2.playVideo()
                time.sleep(2)

    def _get_q_for_position(self, gripper_position):
        iK = self.C.komo_IK(False)
        iK.addObjective(type=ry.OT.sos, feature=ry.FS.qItself, scale=[1] * 16)
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.position, frames=[self.gripper],
                        target=gripper_position, scale=[2e1])
        iK.addObjective(type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1e2])
        iK.optimize()
        self.C.setFrameState(iK.getConfiguration(0))
        via = self.C.getJointState()
        self.C.setFrameState(self.goal_config)
        return np.asarray(via)

    def _get_goal_config(self, move_to=None):
        print("Method: :get_goal_config not implemented for Primtive: ", __name__)
        return

    def _get_komo(self, move_to=None):
        print("Method: :get_komo not implemented for Primtive: ", __name__)
        return

    def _get_joint_interpolation(self, move_to=None):
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
        if self.placing and not self.S.getGripperIsGrasping(self.gripper) and \
                self.place_counter > self.max_place_counter:
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

    def step(self, t):
        i = t - self.t_start
        if i < self.n_steps:
            if self.use_interpolation:
                q = self.q_values[i]
                self.C.setJointState(q)
            else:
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
            if not self.S.getGripperIsGrasping(self.gripper) and not self.is_in_world:
                self.C.attach("world", self.goal)
                self.is_in_world = True
            self.place_counter = self.place_counter + 1
            self.C.setJointState(self.S.get_q())
            self.S.step([], self.tau, ry.ControlMode.none)
        else:
            print("this condition should really not happen, did you forget to define a transition?")
        if not t % 10:
            self.V.setConfiguration(self.C)


class GravComp(Primitive):
    """
    Special class for holding the current position, waiting for an event to happen
    """

    def __init__(self, C, S, V, tau, n_steps, vis=False):
        Primitive.__init__(self, "grav_comp", C, S, V, tau, n_steps,
                           grasping=False, holding=False, placing=False, vis=vis)

    def create_primitive(self, t_start, gripper, goal, move_to=None):
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


class TopGrasp(Primitive):

    def __init__(self, C, S, V,tau, n_steps, interpolation=False, vis=False):
        Primitive.__init__(self, "top_grasp", C, S, V, tau, n_steps,
                           grasping=True, holding=False, placing=False, interpolation=interpolation, vis=vis)

        self.start_overhead_pos = None
        self.needs_start_overhead = None
        self.goal_overhead_pos = None
        self.delta_overhead = 1.4

    def create_primitive(self, t_start, gripper, goal, move_to=None):
        self.start_overhead_pos = self.C.frame(gripper).getPosition()
        self.start_overhead_pos[2] = self.start_overhead_pos[2] + self.delta_overhead

        # check if we need to move the gripper up, before moving to the goal
        self.needs_start_overhead = self.delta_overhead > self.C.frame(gripper).getPosition()[2]
        print("WE NEED Overhead start??", self.needs_start_overhead)
        print(self.delta_overhead)
        print(self.C.frame(gripper).getPosition()[2])

        self.goal_overhead_pos = self.C.frame(goal).getPosition()
        self.goal_overhead_pos[2] = self.goal_overhead_pos[2] + self.delta_overhead

        Primitive.create_primitive(self, t_start, gripper, goal, move_to)

    def _get_goal_config(self, move_to=None):
        block_size =self.C.frame(self.goal).getSize()
        iK = self.C.komo_IK(False)
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.positionRel, frames=[self.goal, self.gripper],
                        target=[0.0, 0.0, -(0.04 + block_size[2]/2)], scale=[2e2])
        #iK.addObjective(type=ry.OT.sos, feature=ry.FS.positionDiff, frames=[self.goal, self.gripper],
        #                scale=[2])
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductZZ, frames=[self.gripper, self.goal],
                        target=[1], scale=[1])
        if block_size[0] > block_size[1]:
            iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductXY,
                            frames=[self.gripper, self.goal], target=[1], scale=[1])
            iK.addObjective(type=ry.OT.eq, feature=ry.qItself,
                            target=[block_size[1]/2]*16, scale=self.mask_gripper)
        else:
            iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductXX,
                            frames=[self.gripper, self.goal], target=[1], scale=[1])
            iK.addObjective(type=ry.OT.eq, feature=ry.qItself,
                            target=[block_size[0]/2] * 16, scale=self.mask_gripper)
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
        if self.needs_start_overhead:
            komo.addObjective(time=[0.0, 0.2], type=ry.OT.eq, feature=ry.FS.position, frames=[self.gripper],
                              target=[0, 0, 0.8], scale=[1e3] * 3, order=2)
        komo.addObjective(time=[0.8, 1.], type=ry.OT.sos, feature=ry.FS.position, frames=[self.gripper],
                          target=[0, 0, -0.8], scale=[1e3] * 3, order=2)
        komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1e3])
        return komo

    def _get_joint_interpolation(self, move_to=None):

        # first via point
        lift_of = self.C.frame(self.gripper).getPosition()
        lift_of[2] = lift_of[2] + 0.2
        if self.needs_start_overhead:
            iK = self.C.komo_IK(False)
            iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductZZ, frames=[self.gripper, self.goal], target=[1],
                            scale=[1])
            iK.addObjective(type=ry.OT.eq, feature=ry.FS.position, frames=[self.gripper],
                            target=lift_of, scale=[2e2])
            iK.addObjective(type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1e2])
            iK.optimize()
            self.C.setFrameState(iK.getConfiguration(0))
            q_via1 = self.C.getJointState()
            q_via1[7] = self.q_start[7]
            q_via1[-1] = self.q_start[-1]
            self.C.setFrameState(self.goal_config)

        # second via point
        overhead_place = self.C.frame(self.gripper).getPosition()
        overhead_place[2] = overhead_place[2] + 0.3

        block_size = self.C.frame(self.goal).getSize()
        iK = self.C.komo_IK(False)
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.positionRel, frames=[self.goal, self.gripper],
                        target=[0.0, 0.0, -(0.2 + block_size[2] / 2)], scale=[2e2])
        # iK.addObjective(type=ry.OT.sos, feature=ry.FS.positionDiff, frames=[self.goal, self.gripper],
        #                scale=[2])
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductZZ, frames=[self.gripper, self.goal], target=[1],
                        scale=[1])
        if block_size[0] > block_size[1]:
            iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductXY,
                            frames=[self.gripper, self.goal], target=[1], scale=[1])
        else:
            iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductXX,
                            frames=[self.gripper, self.goal], target=[1], scale=[1])
        # no contact
        iK.addObjective(type=ry.OT.sos, feature=ry.FS.accumulatedCollisions, scale=[1e2])
        iK.optimize()

        self.C.setFrameState(iK.getConfiguration(0))
        if self.vis:
            self.V.setConfiguration(self.C)
            print("showing via point")
            time.sleep(5)

        q_via2 = self.C.getJointState()
        q_via2[7], q_via2[-1] = self.q_start[7], self.q_start[-1]
        self.C.setFrameState(self.goal_config)

        values = []

        if self.needs_start_overhead:
            print(self.q_start[7], self.q_start[-1])
            print(q_via1[7], q_via1[-1])
            print(self.S.getGripperWidth(self.gripper))
            width = self.S.getGripperWidth(self.gripper)
            phases = [0.2, 0.5, 0.3]
            q_points = [self.q_start, q_via1, q_via2, self.q_goal]
            bezier_profiles = ["EaseInSine", "Linear", "EaseOutSine"]

            for i, (phase, bezier_profile) in enumerate(zip(phases, bezier_profiles)):
                delta = q_points[i + 1] - q_points[i]
                # create n steps between 0 and 1
                steps = np.linspace(0, 1, int(phase * self.n_steps))
                # create bezier
                bezier = beziers.create_bezier(bezier_profile)
                # return list of values
                values = values + [q_points[i] + delta * bezier.solve(t) for t in steps]
        else:
            phases = [0.8, 0.2]
            bezier_profiles = ["EaseInSine", "EaseOutSine"]
            q_points = [self.q_start, q_via2, self.q_goal]
            for i, (phase, bezier_profile) in enumerate(zip(phases, bezier_profiles)):
                delta = q_points[i + 1] - q_points[i]
                # create n steps between 0 and 1
                steps = np.linspace(0, 1, int(phase * self.n_steps))
                # create bezier
                bezier = beziers.create_bezier(bezier_profile)
                # return list of values
                values = values + [q_points[i] + delta * bezier.solve(t) for t in steps]

        return values


class TopPlace(Primitive):

    def __init__(self, C, S, V, tau, n_steps, interpolation=False, vis=False):
        Primitive.__init__(self, __name__, C, S, V, tau, n_steps,
                           grasping=False, holding=False, placing=True, interpolation=interpolation, vis=vis)

    def _get_goal_config(self, move_to=None):
        # get current joint state
        iK = self.C.komo_IK(False)
        block_size = self.C.frame(self.goal).getSize()
        tower_placment = move_to
        tower_placment[2] = tower_placment[2] + block_size[2]/2
        iK.addObjective(type=ry.OT.sos, feature=ry.FS.qItself, target=self.q_start, scale=self.mask_gripper)
        #iK.addObjective(type=ry.OT.eq, feature=ry.FS.vectorX, frames=[self.gripper], target=[0, 1, 0])
        # we assume the object is attached to the frame of the gripper, therefore we can simply just
        # tell the goal object should have a position
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.position,
                        frames=[self.goal], scale=[1e1] * 3, target=tower_placment)
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
        komo.addObjective(time=[], type=ry.OT.sos, feature=ry.FS.qItself, scale=np.asarray(self.mask_gripper)*10,
                          target=self.q_start)
        komo.addObjective(time=[1.], type=ry.OT.eq, feature=ry.FS.qItself,
                          target=self.q_goal, scale=[1e1] * 16)

        komo.addObjective(time=[0.0, 0.2], type=ry.OT.sos, feature=ry.FS.position, frames=[self.gripper],
                          target=[0, 0, 0.8], scale=[1e3] * 3, order=2)
        komo.addObjective(time=[0.8, 1.], type=ry.OT.sos, feature=ry.FS.position, frames=[self.gripper],
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
        move_to[2] =move_to[2] + 0.2

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
        q_via2[7], q_via2[-1] = self.q_start[7], self.q_start[-1]
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

        values = []
        for i, (phase, bezier_profile) in enumerate(zip(phases,bezier_profiles)):
            delta = q_points[i+1] - q_points[i]
            # create n steps between 0 and 1
            steps = np.linspace(0, 1, int(phase*self.n_steps))
            # create bezier
            bezier = beziers.create_bezier(bezier_profile)
            # return list of values
            values = values + [q_points[i] + delta * bezier.solve(t) for t in steps]
        return values


class SideGrasp(Primitive):

    def __init__(self, C, S, V, tau, n_steps, vis=False):
        Primitive.__init__(self, "side_grasp", C, S, V, tau, n_steps,
                           grasping=True, holding=False, placing=False, vis=vis)

    def _get_goal_config(self, move_to=None):
        # generate self.goal configuration
        iK = self.C.komo_IK(False)
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.positionRel, frames=[self.goal, self.gripper], target=[0.0, 0.0, -0.07])
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductXX, frames=[self.goal, self.gripper], target=[-1], scale=[1e2])
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
        komo.addObjective(time=[0.8, 1.], type=ry.OT.eq, feature=ry.FS.scalarProductYZ, frames=[self.goal, self.gripper],
                          target=[-1], scale=[1e3])
        komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.distance, frames=[self.goal, self.gripper], scale=[1e3])
        komo.optimize()
        return komo


class LiftUp(Primitive):
    
    def __init__(self, C, S, V, tau, n_steps, vis=False):
        Primitive.__init__(self,"lift_up", C, S, V, tau, n_steps,
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


class AlignPush(Primitive):

    def __init__(self, C, S, V, tau, n_steps, vis=False):
        Primitive.__init__(self, "align_push", C, S, V, tau, n_steps,
                           grasping=False, holding=False, placing=False, vis=vis)

    def _get_goal_config(self, move_to=None):
        iK = self.C.komo_IK(False)
        # get new position
        q = self.C.getJointState()
        #iK.addObjective(type=ry.OT.sos, feature=ry.FS.qItself, target=q, scale=self.mask_gripper)
        iK.addObjective(type=ry.OT.sos, feature=ry.FS.positionRel, frames=[self.gripper, self.goal],
                        target=[-0.15, 0, 0.01], scale=[1]*3)
        iK.addObjective(type=ry.OT.sos, feature=ry.FS.positionDiff, frames=[self.gripper, self.goal],
                        scale=[0, 0, 1])
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.vectorZ, frames=[self.gripper], target=[0, 0, 1])
        iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductXY, frames=[self.gripper, self.goal], target=[1],
                        scale=[1])
        iK.addObjective(type=ry.OT.ineq, feature=ry.FS.distance, frames=[self.goal, self.gripper],
                          scale=[1e3])
        # no contact
        iK.addObjective(type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions)
        iK.optimize()

        return iK.getConfiguration(0)

    def _get_komo(self, move_to=None):
        komo = self.C.komo_path(1, self.n_steps, self.duration, True)
        komo.addObjective(time=[], type=ry.OT.sos, feature=ry.FS.qItself, scale=[1e1] * 16, order=2)
        komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.jointLimits)
        komo.addObjective(time=[1.], type=ry.OT.eq, feature=ry.FS.qItself, target=self.q_goal,
                          scale=[1e2] * 16)
        komo.addObjective(time=[], type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[1])
        komo.optimize()
        return komo
