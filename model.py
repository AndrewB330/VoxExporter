from functools import reduce
from typing import Optional, List, Tuple

import numpy as np

from bytes_reader import BytesReader
from utils import decode_rotation, decode_translation


class AABB:
    """
    Axis-aligned bounding box (AABB)
    defined by two diagonal points with minimal and maximal coordinates values
    """

    def __init__(self, min_coords: np.ndarray, max_coords: np.ndarray):
        self.min_coords = min_coords
        self.max_coords = max_coords

    def transformed(self, translate: np.ndarray, rotate: np.ndarray) -> 'AABB':
        coords = np.vstack([(rotate @ self.min_coords.T + translate), (rotate @ self.max_coords.T + translate)])
        return AABB(coords.min(axis=0), coords.max(axis=0))

    def union(self, other: 'AABB') -> 'AABB':
        coords = np.vstack([self.min_coords, self.max_coords, other.min_coords, other.max_coords])
        return AABB(coords.min(axis=0), coords.max(axis=0))

    def get_size(self) -> np.ndarray:
        return self.max_coords - self.min_coords

    def __str__(self) -> str:
        return f'AABB[min {self.min_coords} -> max {self.max_coords}]'


class Model:
    def __init__(self, x_size: int, y_size: int, z_size: int, voxels: List[Tuple[int, int, int, int]]):
        self.x_size, self.y_size, self.z_size = x_size, y_size, z_size
        self.voxels_map = np.zeros((x_size, y_size, z_size)).astype(np.uint8)
        self.origin = np.array([x_size // 2, y_size // 2, z_size // 2]).astype(int)
        for x, y, z, c in voxels:
            self.voxels_map[x, y, z] = c
        self.cached_aabb: Optional[AABB] = None

    def get_size(self):
        return self.x_size, self.y_size, self.z_size

    @staticmethod
    def from_bytes(size_bytes: bytes, voxels_bytes: bytes):
        x_size, y_size, z_size = BytesReader(size_bytes).read_ints(3)
        reader = BytesReader(voxels_bytes)
        n = reader.read_int()
        voxels = [reader.read_bytes(4) for _ in range(n)]
        return Model(x_size, y_size, z_size, voxels)

    def transposed(self, t: tuple = (0, 2, 1)) -> 'Model':
        size = self.get_aabb().get_size()[np.array(t)]
        m = Model(size[0], size[1], size[2], [])
        m.origin = self.origin[np.array(t, dtype=int)]
        m.voxels_map = self.voxels_map.transpose(t)
        m.cached_aabb = None
        return m

    def get_aabb(self) -> AABB:
        if self.cached_aabb is not None:
            return self.cached_aabb
        corner = np.array([self.x_size, self.y_size, self.z_size])
        self.cached_aabb = AABB(np.zeros(3, dtype=int), corner).transformed(-self.origin, np.identity(3, dtype=int))
        return self.cached_aabb

    def get_at(self, pos: np.ndarray, direction: np.ndarray = np.ones(3, dtype=int)):
        pos = pos + (direction - np.ones(3, dtype=int)) // 2  # todo: describe why?
        if np.any(pos >= self.get_aabb().max_coords):
            return 0
        if np.any(pos < self.get_aabb().min_coords):
            return 0
        x, y, z = pos + self.origin
        return self.voxels_map[x, y, z]


class Node:
    def __init__(self, node_id):
        self.node_id = node_id

    def get_aabb(self) -> AABB:
        return AABB(np.zeros(3), np.zeros(3))

    def get_at(self, pos: np.ndarray, direction: np.ndarray) -> int:
        return 0


class Transform(Node):
    def __init__(self, node_id, child_id, rotation, translation):
        super().__init__(node_id)
        self.child_id = child_id
        self.rotation = rotation.astype(int)
        self.translation = translation.astype(int)
        self.inverse_rotation = np.linalg.inv(self.rotation).astype(int)
        self.child: Optional[Node] = None

    @staticmethod
    def from_bytes(transform_bytes):
        reader = BytesReader(transform_bytes)
        node_id = reader.read_int(signed=True)
        attr = reader.read_dict()
        child_id = reader.read_int(signed=True)

        assert (reader.read_int(signed=True) == -1)
        layer_id = reader.read_int(signed=True)
        assert (reader.read_int() == 1)

        frame_attr = reader.read_dict()

        rotation = decode_rotation(0 if '_r' not in frame_attr else int(frame_attr['_r']))
        translation = decode_translation('0 0 0' if '_t' not in frame_attr else frame_attr['_t'])
        return Transform(node_id, child_id, rotation, translation)

    def get_aabb(self) -> AABB:
        return self.child.get_aabb().transformed(self.translation, self.rotation)

    def get_at(self, pos: np.ndarray, direction: np.ndarray) -> int:
        return self.child.get_at(self.inverse_rotation @ (pos - self.translation), self.inverse_rotation @ direction)


class Group(Node):
    def __init__(self, node_id, children_ids):
        super().__init__(node_id)
        self.children_ids = children_ids
        self.children = []

    @staticmethod
    def from_bytes(group_bytes):
        reader = BytesReader(group_bytes)
        node_id = reader.read_int(signed=True)
        attr = reader.read_dict()
        children_ids = [reader.read_int(signed=True) for _ in range(reader.read_int())]
        return Group(node_id, children_ids)

    def get_aabb(self) -> AABB:
        return reduce(lambda a, b: a.union(b), map(lambda a: a.get_aabb(), self.children))

    def get_at(self, pos: np.ndarray, direction: np.ndarray) -> int:
        colors = list(filter(lambda c: c > 0, map(lambda node: node.get_at(pos, direction), self.children)))
        return 0 if len(colors) == 0 else colors[0]


class Shape(Node):
    def __init__(self, node_id, model_id):
        super().__init__(node_id)
        self.model_id = model_id
        self.model = None

    @staticmethod
    def from_bytes(shape_bytes):
        reader = BytesReader(shape_bytes)
        node_id = reader.read_int(signed=True)
        attr = reader.read_dict()
        assert (reader.read_int() == 1)
        model_id = reader.read_int(signed=True)
        return Shape(node_id, model_id)

    def get_aabb(self) -> AABB:
        return self.model.get_aabb()

    def get_at(self, pos: np.ndarray, direction: np.ndarray) -> int:
        return self.model.get_at(pos, direction)


def merge_models(root: Node):
    aabb = root.get_aabb()
    voxels = []
    for x in range(aabb.min_coords[0], aabb.max_coords[0]):
        for y in range(aabb.min_coords[1], aabb.max_coords[1]):
            for z in range(aabb.min_coords[2], aabb.max_coords[2]):
                color = root.get_at(np.array([x, y, z]), np.ones(3, dtype=int))
                if color > 0:
                    voxels.append((x - aabb.min_coords[0], y - aabb.min_coords[1], z - aabb.min_coords[2], color))
    x_size, y_size, z_size = aabb.get_size()
    model = Model(x_size, y_size, z_size, voxels)
    model.origin = -aabb.min_coords
    return model
