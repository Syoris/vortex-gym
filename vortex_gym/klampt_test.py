from klampt import WorldModel, Geometry3D
from klampt import vis
from klampt.math import vectorops, so3, se3
from klampt.model.create import primitives
from pathlib import Path

# you will need to change this to the absolute or relative path to Klampt-examples
KLAMPT_EXAMPLES = Path('C:/Users/charl/Local Documents/git/Klampt-examples')

shelf_dims = (0.4, 0.4, 0.3)
shelf_offset_x = 0.8
shelf_offset_y = 0.1
shelf_height = 0.65


def make_shelf(world, width, depth, height, wall_thickness=0.005):
    """Makes a new axis-aligned "shelf" centered at the origin with
    dimensions width x depth x height. Walls have thickness wall_thickness.
    """
    left = Geometry3D()
    right = Geometry3D()
    back = Geometry3D()
    bottom = Geometry3D()
    top = Geometry3D()
    # method 1
    left.loadFile(KLAMPT_EXAMPLES + '/data/objects/cube.off')
    left.transform([wall_thickness, 0, 0, 0, depth, 0, 0, 0, height], [-width * 0.5, -depth * 0.5, 0])
    right.loadFile(KLAMPT_EXAMPLES + '/data/objects/cube.off')
    right.transform([wall_thickness, 0, 0, 0, depth, 0, 0, 0, height], [width * 0.5, -depth * 0.5, 0])
    # method 2
    back.loadFile(KLAMPT_EXAMPLES + '/data/objects/cube.off')
    back.scale(width, wall_thickness, height)
    back.translate([-width * 0.5, depth * 0.5, 0])
    # equivalent to back.transform([width,0,0,0,wall_thickness,0,0,0,height],[-width*0.5,depth*0.5,0])
    # method 3
    bottom = primitives.box(width, depth, wall_thickness, center=[0, 0, 0])
    top = primitives.box(width, depth, wall_thickness, center=[0, 0, height - wall_thickness * 0.5])
    shelfgeom = Geometry3D()
    shelfgeom.setGroup()
    for i, elem in enumerate([left, right, back, bottom, top]):
        g = Geometry3D(elem)
        shelfgeom.setElement(i, g)
    shelf = world.makeTerrain('shelf')
    shelf.geometry().set(shelfgeom)
    shelf.appearance().setColor(0.2, 0.6, 0.3, 1.0)
    return shelf


w = WorldModel()
if not w.readFile('myworld.xml'):
    raise RuntimeError("Couldn't read the world file")

shelf = make_shelf(w, *shelf_dims)
shelf.geometry().translate((shelf_offset_x, shelf_offset_y, shelf_height))

obj = w.makeRigidObject('point_cloud_object')
obj.geometry().loadFile(KLAMPT_EXAMPLES + '/data/objects/apc/genuine_joe_stir_sticks.pcd')
# set up a "reasonable" inertial parameter estimate for a 200g object
m = obj.getMass()
m.estimate(obj.geometry(), 0.200)
obj.setMass(m)
# we'll move the box slightly forward so the robot can reach it
obj.setTransform(so3.identity(), [shelf_offset_x - 0.05, shelf_offset_y - 0.3, shelf_height + 0.01])


vis.add('world', w)
vis.add('ghost', w.robot(0).getConfig(), color=(0, 1, 0, 0.5))
vis.edit('ghost')
from klampt import Simulator

sim = Simulator(w)


def setup():
    vis.show()


def callback():
    sim.controller(0).setPIDCommand(vis.getItemConfig('ghost'), [0] * w.robot(0).numLinks())
    sim.simulate(0.01)
    sim.updateWorld()


vis.loop(setup=setup, callback=callback)
