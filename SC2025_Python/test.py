import ctypes
from misc import *
import numpy as np

dll = ctypes.cdll.LoadLibrary("..\\bin\\SCAPI.dll")
SJCUDARenderer_New = dll.SJCUDARenderer_New
SJCUDARenderer_Initialize = dll.SJCUDARenderer_Initialize
SJCUDARenderer_LoadMPI = dll.SJCUDARenderer_LoadMPI
SJCUDARenderer_InitializeRendering = dll.SJCUDARenderer_InitializeRendering
SJCUDARenderer_Rendering = dll.SJCUDARenderer_Rendering
SJCUDARenderer_Finalize = dll.SJCUDARenderer_Finalize

SJCUDARenderer_New.restype = ctypes.c_void_p
SJCUDARenderer_New.argtypes = []

SJCUDARenderer_Initialize.restype = ctypes.c_int
SJCUDARenderer_Initialize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]

SJCUDARenderer_LoadMPI.restype = ctypes.c_int
SJCUDARenderer_LoadMPI.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ubyte)]

SJCUDARenderer_InitializeRendering.restype = ctypes.c_int
SJCUDARenderer_InitializeRendering.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int]

SJCUDARenderer_Rendering.restype = ctypes.c_int
SJCUDARenderer_Rendering.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ubyte)]

SJCUDARenderer_Finalize.restype = ctypes.c_int
SJCUDARenderer_Finalize.argtypes = [ctypes.c_void_p]


def load_mpi(basedir):
    metadata = os.path.join(basedir, 'metadata.txt')
    mpibinary = os.path.join(basedir, 'mpi.b')
    lines = open(metadata, 'r').read().split('\n')
    h, w, d = [int(x) for x in lines[0].split(' ')[:3]]
    focal = float(lines[0].split(' ')[-1])
    data = np.frombuffer(open(mpibinary, 'rb').read(), dtype=np.uint8)/255.
    data = data.reshape([d,h,w,4]).transpose([1,2,0,3])
    #data.reshape([d, h, w, 4])
    
    data[...,-1] = np.minimum(1., data[...,-1]+1e-8)
    
    pose = np.array([[float(x) for x in l.split(' ')] for l in lines[1:5]]).T
    pose = np.concatenate([pose, np.array([h,w,focal]).reshape([3,1])], -1)
    pose = np.concatenate([-pose[:,1:2], pose[:,0:1], pose[:,2:]], 1)
    idepth, cdepth = [float(x) for x in lines[5].split(' ')[:2]]
    
    return data, idepth, cdepth, pose

scenedir = "..\\data\\Sample"
poses, pts3d, perm, w2c, c2w, hwf = load_colmap_data(scenedir)
cdepth, idepth = computecloseInfinity(poses, pts3d, perm)
close_depth = np.min(cdepth) * 0.9
inf_depth = np.max(idepth) * 2.0
focal = poses[2, 4, :]

