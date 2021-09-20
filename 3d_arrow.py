# %matplotlib notebook
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
from scipy.spatial.transform import Rotation

# def rotation_matrix(d):
#     """
#     Calculates a rotation matrix given a vector d. The direction of d
#     corresponds to the rotation axis. The length of d corresponds to 
#     the sin of the angle of rotation.
#     Variant of: http://mail.scipy.org/pipermail/numpy-discussion/2009-March/040806.html
#     """
#     sin_angle = np.linalg.norm(d)
#     if sin_angle == 0:
#         return np.identity(3)
#     d /= sin_angle
#     eye = np.eye(3)
#     ddt = np.outer(d, d)
#     skew = np.array([[    0,  d[2],  -d[1]],
#         [-d[2],     0,  d[0]],
#         [d[1], -d[0],    0]], dtype=np.float64)
#     M = ddt + np.sqrt(1 - sin_angle**2) * (eye - ddt) + sin_angle * skew
#     return M

def pathpatch_translate(pathpatch, delta):
    """
    Translates the 3D pathpatch by the amount delta.
    """
    pathpatch._segment3d += delta

def _from_normal(old, new):
    vec = np.cross(old, new)
    angle = np.arccos(np.inner(old, new))
    rot_vec = angle * vec/np.linalg.norm(vec)
    rot = Rotation.from_rotvec(rotvec)
setattr(Rotation, 'from_normal', _from_normal)

# def pathpatch_2d_to_3d(pathpatch, z = 0, normal = 'z'):
def pathpatch_2d_to_3d(pathpatch, z = 0, normal = (0,0,1)):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.
    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    # if type(normal) is str: #Translate strings to normal vectors
    #     index = "xyz".index(normal)
    #     normal = np.roll((1.0,0,0), index)
    # normal /= np.linalg.norm(normal) #Make sure the vector is normalised
    #
    path = pathpatch.get_path() #Get the path and the associated transform
    trans = pathpatch.get_patch_transform()
    path = trans.transform_path(path) #Apply the transform
    pathpatch.__class__ = art3d.PathPatch3D #Change the class
    pathpatch._code3d = path.codes #Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor #Get the face color    
    verts = path.vertices #Get the vertices in 2D
    rot = Rotation._from_normal((0,0,1), normal)
    pathpatch._segment3d = np.array([rot.apply((x, y, 0)) + (0, 0, z) for x, y in verts])

class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)
    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

def _arrow3D(ax, pos, normal, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''
    arrow = Arrow3D(pos[0], pos[1], pos[2], normal[0], normal[1], normal[2], mutation_scale=20, arrowstyle="-|>", *args, **kwargs)
    ax.add_artist(arrow)
setattr(Axes3D, 'arrow3D', _arrow3D)

def _origin3D(ax, pos=(0,0,0), z_vec=(0,0,1)):
    x_axis = (1,0,0)
    y_axis = (0,1,0)
    z_axis = (0,0,1)
    # rot = Rotation.from_euler('ZYX', normal, degrees=True)
    rot = Rotation.from_rotvec(np.cross(z_axis, z_vec))
    ax.arrow3D(pos, rot.apply(x_axis), color='r')
    ax.arrow3D(pos, rot.apply(y_axis), color='g')
    ax.arrow3D(pos, rot.apply(z_axis), color='b')
setattr(Axes3D, 'origin3D', _origin3D)

def _circle3D(ax, pos, radius, normal, *args, **kwargs):
    # p = Circle((center[0], center[1]),radius)
    p = Circle((0,0),radius, *args, **kwargs)
    ax.add_patch(p)
    # art3d.pathpatch_2d_to_3d(p, z=center[2], zdir=zdir, *args, **kwargs)
    pathpatch_2d_to_3d(p, z=0, normal=normal)
    pathpatch_translate(p, pos)
setattr(Axes3D, 'circle3D', _circle3D)

##

fig = plt.figure()
ax = fig.gca(projection='3d', xlabel='x', ylabel='y', zlabel='z')
ax.set_xlim(0,8)
ax.set_ylim(0,8)
ax.set_zlim(0,8)

# global
# ax.origin3D(pos=(0,0,0), z_vec=Rotation.from_euler('ZYX',(0, 45, 45),degrees=True).apply((0,0,1)))
ax.origin3D()

# face
face_pos = (2,5,1)
face_z_vec = Rotation.from_euler('ZYX',(0, 45, 0),degrees=True).apply((0,0,1))
ax.origin3D(face_pos, face_z_vec)
ax.circle3D(face_pos, 1, face_z_vec, color='gray', alpha=0.2)

# camera
camera_pos = (5,5,5)
camera_z_vec = (1, 0, 0)
camera_z_vec /= np.linalg.norm(camera_z_vec)
ax.origin3D(camera_pos, camera_z_vec)

# vec = np.array([1,0,0])
# ax.arrow3D((0,0,0), vec)
# euler = np.array([ 0, 45, 0])
# rot = Rotation.from_euler('ZYX', euler, degrees=True)
# ax.arrow3D((0,0,0), rot.apply(vec))

plt.draw()
plt.show()

