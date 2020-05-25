import libry as ry
from os.path import join

pathRepo = '/home/jason/git/robotics-course/'


def setup_challenge_env(add_red_ball=False, number_objects=30):

    # -- Add REAL WORLD configuration and camera
    R = ry.Config()
    R.addFile(join(pathRepo, "scenarios/challenge.g"))

    if add_red_ball:
        # only add 1 red ball
        number_objects = 2
        # you can also change the shape & size
        R.getFrame("obj0").setColor([1., 0, 0])
        R.getFrame("obj0").setShape(ry.ST.sphere, [.03])
        # RealWorld.getFrame("obj0").setShape(ry.ST.ssBox, [.05, .05, .2, .01])
        R.getFrame("obj0").setPosition([0.0, .05, 2.])
        R.getFrame("obj0").setContact(1)

        R.getFrame("obj1").setColor([0, 0, 1.])
        R.getFrame("obj1").setShape(ry.ST.sphere, [.03])
        # RealWorld.getFrame("obj0").setShape(ry.ST.ssBox, [.05, .05, .2, .01])
        R.getFrame("obj1").setPosition([0.0, .3, 2.])
        R.getFrame("obj1").setContact(1)

    for o in range(number_objects, 30):
        name = "obj%i" % o
        print("deleting", name)
        R.delFrame(name)

    S = R.simulation(ry.SimulatorEngine.physx, True)

    # Change color of objects
    S.addSensor("camera")

    C = ry.Config()
    C.addFile(join(pathRepo, "scenarios/pandasTable.g"))
    V = ry.ConfigurationViewer()
    V.setConfiguration(C)
    C.addFrame("goal")
    C.addFrame("goal2")

    return R, S, C, V

def setup_camera(C, f):
    # setup camera
    cameraFrame = C.frame("camera")
    # the focal length
    f = 0.895
    f = f * 360.
    fxfypxpy = [f, f, 320., 180.]

    return cameraFrame, fxfypxpy

