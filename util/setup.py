import random
import libry as ry
import util.perception as perc
from os.path import join
from util.path_to_repo import path_to_rai
import numpy as np


def setup_challenge_env(add_red_ball=False, number_objects=30, show_background=False):
    # -- Add empty REAL WORLD configuration and camera
    # R = ry.Config()
    # R.addFile(join(pathRepo, "scenarios/pandasTable.g"))
    # S = R.simulation(ry.SimulatorEngine.physx, True)
    # S.addSensor("camera")

    # back_frame = perc.extract_background(S, duration=2, vis=show_background)
    back_frame = None

    R = ry.Config()
    R.addFile(join(path_to_rai, "scenarios/challenge.g"))

    if add_red_ball:
        # only add 1 red ball
        number_objects = 1
        # you can also change the shape & size
        R.getFrame("obj0").setColor([1., 0, 0])
        R.getFrame("obj0").setShape(ry.ST.sphere, [0.03])
        # RealWorld.getFrame("obj0").setShape(ry.ST.ssBox, [.05, .05, .2, .01])
        R.getFrame("obj0").setPosition([0.0, -.02, 0.68])
        R.getFrame("obj0").setContact(1)

        # R.getFrame("obj1").setColor([0, 0, 1.])
        # R.getFrame("obj1").setShape(ry.ST.sphere, [.03])
        # # RealWorld.getFrame("obj0").setShape(ry.ST.ssBox, [.05, .05, .2, .01])
        # R.getFrame("obj1").setPosition([0.0, .3, 2.])
        # R.getFrame("obj1").setContact(1)

    for o in range(number_objects, 30):
        name = "obj%i" % o
        R.delFrame(name)

    # C, S, V = _get_CSV(R)

    return R, back_frame  # S, C, V,


def setup_env_subgoal_1(show_background=False):
    num_blocks = 2
    R, back_frame = setup_challenge_env(False, num_blocks, show_background=show_background)

    # side = 0.13
    # positions = [
    #     [0.3, .3, 0.65 + side / 2],
    #     [0.6, 0.2, 0.65 + side / 2],
    #     # [0.0, -.2, 0.65+side/2],
    #     # [-0.1, 0, 0.65+side/2],
    #     # [0.1, 0, 0.65+side/2],
    # ]
    # for o in range(num_blocks):
    #     name = "obj%i" % o
    #     box = R.frame(name)
    #     box.setPosition(positions[o])
    #     box.setColor([1, 0, 0])
    #     box.setShape(ry.ST.box, size=[side, side, side, 0.001])
    #     box.setQuaternion([1, 0, 0, 0])
    #     box.setContact(1)
    box1 = R.frame("obj0")
    box1.setShape(ry.ST.ssBox, size=[0.12, 0.12, 0.1, 0.0001])
    box1.setPosition([0.25, 0.1, 0.7+0.05])
    box1.setColor([1, 1, 0])
    box1.setQuaternion([1, 0, 0, 0])
    box1.setContact(1)

    box2 = R.frame("obj1")
    box2.setShape(ry.ST.ssBox, size=[0.12, 0.12, 0.1, 0.0001])
    box2.setPosition([-0.4, 0.2, 0.7+ 0.05])
    box2.setColor([0.5, 0, 1])
    box2.setQuaternion([1, 0, 0, 0])
    box2.setContact(1)

    # table = R.frame("table_g")
    # table.setShape(ry.ST.ssBox, size=[0.3, 0.3, 0.01, 0])
    # table.setPosition([0.0, -0.4, .69])
    # table.setColor([1, 1, 1])
    # table.setContact(1)

    C, S, V = _get_CSV(R)

    table = C.addFrame("table_g")
    table.setShape(ry.ST.ssBox, size =[0.3, 0.3, 0.01, 0])
    table.setPosition([0.0, -0.4, .66])
    table.setColor([0, 0, 0])
    V.setConfiguration(C)

    return R, S, C, V, back_frame


def setup_env_subgoal_2(show_background=False):
    num_blocks = 5
    R, back_frame = setup_challenge_env(False, num_blocks, show_background=show_background)

    color = [[0, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0],
             [1, 0.5, 0], [0.5, 0, 1], [0, 1, 0.5], [0, 0.5, 1], [0.5, 1, 0]]

    side = 0.13
    positions = [
        [0.3, .3, 0.7 + side / 2],
        [-0.5, .2, 0.7 + side / 2],
        [-0.3, .2, 0.7 + side / 2],
        [0.5, .15, 0.7 + side / 2],
        [0.6, 0.3, 0.7 + side / 2],
    ]
    for i, o in enumerate(range(num_blocks)):
        name = "obj%i" % o
        box = R.frame(name)
        box.setPosition(positions[o])
        box.setShape(ry.ST.ssBox, size=[side - i * 0.01, side - i * 0.01, side - i * 0.01, 0.01])
        box.setQuaternion([1, 0, 0, 0])
        box.setContact(1)
        box.setColor(color[i])

    C, S, V = _get_CSV(R)

    return R, S, C, V, back_frame


def setup_env_subgoal_3(show_background=False):
    num_blocks = 2
    R, back_frame = setup_challenge_env(False, num_blocks, show_background=show_background)

    s = 0.13
    color = [[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 0],
             [1, 0.5, 0], [0.5, 0, 1], [0, 1, 0.5], [0, 0.5, 1], [0.5, 1, 0]]

    positions = [
        [0.6, 0.2, 0.65 + 0.1 / 2],
        # [-0.1, .2, 0.65+side/2],
        # [-0.2, -.1, 0.65+side/2],
        # [0.5, .15, 0.65+side/2],
        [-0.6, 0.3, 0.65 + 0.13 / 2]
    ]

    sizes = [
        [0.3, 0.3, 0.1, 0.0001],
        # [-0.1, .2, 0.65+side/2],
        # [-0.2, -.1, 0.65+side/2],
        # [0.5, .15, 0.65+side/2],
        [0.13, 0.13, 0.1, 0.0001],
    ]
    for i in range(num_blocks):
        name = "obj%i" % i
        box = R.frame(name)
        box.setPosition(positions[i])
        box.setShape(ry.ST.ssBox, sizes[i])
        box.setQuaternion([1, 0, 0, 0])
        box.addAttribute("friction", 1.0)
        box.setColor(color[i])
        box.setContact(1)

    C, S, V = _get_CSV(R)

    return R, S, C, V, back_frame


def setup_env_subgoal_4(show_background=False):
    num_blocks = 5
    R, back_frame = setup_challenge_env(False, num_blocks, show_background=show_background)

    s = 0.13
    color = [[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 0],
             [1, 0.5, 0], [0.5, 0, 1], [0, 1, 0.5], [0, 0.5, 1], [0.5, 1, 0]]

    s = 0.15
    positions = [
        [0.6, 0.1, 0.7 + 0.1 / 2],
        [-0.6, 0.05, 0.7 + 0.1 / 2],
        [0.1, 0.3, 0.7 + 0.15 / 2],
        [0.5, 0.25, 0.7 + 0.14 / 2],
        [-0.6, 0.3, 0.7 + 0.13 / 2]
    ]

    sizes = [
        [0.3, 0.3, 0.1, 0],
        [0.3, 0.3, 0.1, 0],
        [0.15, 0.15, 0.15, 0],
        [0.14, 0.14, 0.14, 0],
        [0.13, 0.13, 0.13, 0],

    ]
    for i in range(num_blocks):
        name = "obj%i" % i
        box = R.frame(name)
        box.setPosition(positions[i])
        box.setShape(ry.ST.ssBox, sizes[i])
        box.setQuaternion([1, 0, 0, 0])
        box.addAttribute("friction", 1.0)
        box.setContact(1)
        box.setColor(color[i])
        box.setMass(10000000)

    C, S, V = _get_CSV(R)

    return R, S, C, V, back_frame


def setup_env_subgoal_4(show_background=False):
    num_blocks = 5
    R, back_frame = setup_challenge_env(False, num_blocks, show_background=show_background)

    s = 0.13
    color = [[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 0],
             [1, 0.5, 0], [0.5, 0, 1], [0, 1, 0.5], [0, 0.5, 1], [0.5, 1, 0]]

    s = 0.15
    positions = [
        [0.6, 0.1, 0.7 + 0.1 / 2],
        [-0.6, 0.05, 0.7 + 0.1 / 2],
        [0.1, 0.3, 0.7 + 0.15 / 2],
        [0.5, 0.25, 0.7 + 0.14 / 2],
        [-0.6, 0.3, 0.7 + 0.13 / 2]
    ]

    sizes = [
        [0.3, 0.3, 0.1, 0],
        [0.3, 0.3, 0.1, 0],
        [0.15, 0.15, 0.15, 0],
        [0.14, 0.14, 0.14, 0],
        [0.13, 0.13, 0.13, 0],

    ]
    for i in range(num_blocks):
        name = "obj%i" % i
        box = R.frame(name)
        box.setPosition(positions[i])
        box.setShape(ry.ST.ssBox, sizes[i])
        box.setQuaternion([1, 0, 0, 0])
        box.addAttribute("friction", 1.0)
        box.setContact(1)
        box.setColor(color[i])
        box.setMass(10000000)

    C, S, V = _get_CSV(R)

    return R, S, C, V, back_frame



def setup_env_max_stack(show_background=False):
    n_row = 3
    n_col = 2

    num_blocks = n_row*n_col


    R, back_frame = setup_challenge_env(False, num_blocks*2, show_background=show_background)

    color = [[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 0],
             [1, 0.5, 0], [0.5, 0, 1], [0, 1, 0.5], [0, 0.5, 1], [0.5, 1, 0]]*2

    side = 0.15
    space = 0.3

    positions = np.zeros((num_blocks, 3))

    pos = 0
    for i in range(n_row):
        for j in range(n_col):
            positions[pos, :] = np.array([0.1 + i*space, 0. + j * space, 0.7+ (side - pos * 0.005)/2])
            pos += 1

    positions2 = np.array(positions, copy=True)
    positions2[:, 0] = -1 * positions2[:, 0]

    positions = np.concatenate((positions, positions2), axis=0)

    print(positions)

    for i in range(num_blocks*2):
        name = "obj%i" % i
        box = R.frame(name)
        box.setPosition(positions[i, :])

        box.setQuaternion([1, 0, 0, 0])
        box.setShape(ry.ST.box, size=[side - i * 0.005, side - i * 0.005, side - i * 0.005]) # 0.001
        box.addAttribute("friction", 1.0)
        box.setContact(1)
        box.setColor(color[i])
        box.setMass(10000000)

    C, S, V = _get_CSV(R)

    return R, S, C, V, back_frame

"""
Environment where each object has different colors, for better object recognition in perception
"""


def setup_color_challenge_env():
    num_blocks = 4
    # random.seed(10)

    R = ry.Config()

    R.addFile(join(path_to_rai, "scenarios/challenge.g"))

    # positions = [
    #     [0.02, 0.23, 0.7],
    #     [-0.35,-0.1, 0.7],
    #     [0.2, 0.45, 0.7],
    #     # [0.5, .15, 0.65+side/2],
    #     [0.0, -0.1, 0.7]
    # ]
    #
    # sizes = [
    #     [0.12, 0.08, 0.10, 0],
    #     [0.17, 0.09, 0.12, 0],
    #     [0.14, 0.10, 0.12, 0],
    #     # [0.5, .15, 0.65+side/2],
    #     [0.15, 0.20, 0.11, 0],
    # ]
    #
    # color = [[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0],
    #          [1, 0.5, 0], [0.5, 0, 1], [0, 1, 0.5], [0, 0.5, 1], [0.5, 1, 0]]
    #
    # for i in range(num_blocks):
    #     name = "obj%i" % i
    #     box = R.frame(name)
    #     box.setPosition(positions[i])
    #     box.setColor([0, 0, 1])
    #     box.setShape(ry.ST.ssBox, sizes[i])
    #     box.setQuaternion([1, 0, 0, 0])
    #     box.setContact(1)

    # Change color of objects depending how many objects in .g file are
    obj_count = 0
    for n in R.getFrameNames():
        if n.startswith("obj"):
            obj_count += 1

    for o in range(0, obj_count):
        color = [[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 0],
                 [1, 0.5, 0], [0.5, 0, 1], [0, 1, 0.5], [0, 0.5, 1], [0.5, 1, 0]]
        name = "obj%i" % o
        R.frame(name).setContact(1)
        R.frame(name).setColor(color[o])

    S = R.simulation(ry.SimulatorEngine.physx, True)
    S.addSensor("camera")

    C = ry.Config()
    C.addFile(join(path_to_rai, 'scenarios/pandasTable.g'))
    V = ry.ConfigurationViewer()
    V.setConfiguration(C)

    R_gripper = C.frame("R_gripper")
    R_gripper.setContact(1)
    L_gripper = C.frame("L_gripper")
    L_gripper.setContact(1)

    return R, S, C, V


def setup_env_test_edge_grasp(show_background=False):
    num_blocks = 1
    R, back_frame = setup_challenge_env(False, num_blocks, show_background=show_background)

    height = 0.08
    width = 0.3
    length = width
    position = [0.9, 0.1, 0.65 + height / 2]

    box = R.frame("obj0")
    box.setPosition(position)
    box.setColor([1, 0, 0])
    box.setShape(ry.ST.ssBox, size=[length, width, height, 0])
    box.setQuaternion([1, 0, 0, 0])
    box.addAttribute("friction", 1.0)
    box.setContact(1)

    C, S, V = _get_CSV(R)

    return R, S, C, V, back_frame


def _get_CSV(R):
    S = R.simulation(ry.SimulatorEngine.physx, True)
    S.addSensor("camera")
    C = ry.Config()
    C.addFile(join(path_to_rai, "scenarios/pandasTable.g"))
    V = ry.ConfigurationViewer()
    V.setConfiguration(C)
    return C, S, V


def setup_camera(C):
    # setup camera
    cameraFrame = C.frame("camera")
    # the focal length
    f = 0.895
    f = f * 360.
    fxfypxpy = [f, f, 320., 180.]

    return cameraFrame, fxfypxpy
