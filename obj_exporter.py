import os
from enum import Enum
from queue import Queue
from typing import List

import numpy as np
from PIL import Image

from model import Model
from utils import projection_inv, projection, get_slice, pack, STEPS_2D, fill_color_map

BASIS = np.identity(3, dtype=int)


class Direction(Enum):
    LEFT = 0
    RIGHT = 1
    DOWN = 2
    UP = 3
    BACK = 4
    FORWARD = 5


ALL_DIRECTIONS = [Direction.LEFT, Direction.RIGHT, Direction.DOWN, Direction.UP, Direction.BACK, Direction.FORWARD]


def axis_by_direction(direction: Direction):
    if direction.value < 2:
        return 0
    if direction.value < 4:
        return 1
    return 2


def sign_by_direction(direction: Direction):
    return -1 if direction.value % 2 == 0 else +1


def normal_by_direction(direction: Direction):
    return BASIS[axis_by_direction(direction)] * sign_by_direction(direction)


def get_components(slice_):
    shape = slice_.shape

    ids = np.zeros(slice_.shape)
    counter = 0

    min_x, max_x, min_y, max_y = 0, 0, 0, 0

    def fill(x, y):
        nonlocal min_x, max_x, min_y, max_y
        q = Queue()
        q.put((x, y))
        size_limit = 128
        while not q.empty():
            cx, cy = q.get()
            for dx, dy in STEPS_2D:
                nx, ny = cx + dx, cy + dy
                if max(nx - min_x, max_x - nx, ny - min_y, max_y - ny) > size_limit:
                    continue
                if 0 <= nx < shape[0] and 0 <= ny < shape[1] and slice_[nx][ny] != 0 and ids[nx][ny] == 0:
                    min_x, max_x, min_y, max_y = min(min_x, nx), max(max_x, nx), min(min_y, ny), max(max_y, ny)
                    ids[nx][ny] = counter
                    q.put((nx, ny))

    components = []

    for x in range(shape[0]):
        for y in range(shape[1]):
            if slice_[x, y] > 0 and ids[x, y] == 0:
                counter += 1
                ids[x, y] = counter
                min_x, max_x, min_y, max_y = x, x, y, y
                fill(x, y)
                components.append((ids == counter, (min_x, max_x, min_y, max_y)))

    return components


def get_decomposition(mask, min_x, max_x, min_y, max_y):
    cum_sum = np.cumsum(mask, axis=1)
    rectangles = []
    for i in range(min_x, max_x + 1):
        for j in range(min_y, max_y + 1):
            if not mask[i, j]:
                continue
            w, h = 1, 1
            while j + h < mask.shape[1] and mask[i, j + h]:
                h += 1
            while i + w < mask.shape[0] and (cum_sum[i + w, j + h - 1] - (cum_sum[i + w, j - 1] if j > 0 else 0)) == h:
                w += 1
            mask[i:i + w, j:j + h] = False
            rectangles.append((i, i + w - 1, j, j + h - 1))
    return rectangles


class Part:
    def __init__(self, data, aabb, direction, offset):
        self.data = data
        self.aabb = aabb
        self.direction = direction
        self.offset = offset
        self.rectangles = get_decomposition(data != 0, *aabb)
        self.uv_offset = (0, 0)
        self.transposed = False

    def get_data_inside_aabb(self):
        return self.data[self.aabb[0]:self.aabb[1] + 1, self.aabb[2]:self.aabb[3] + 1]

    def get_faces(self):
        u_offset, v_offset = self.uv_offset
        axis = axis_by_direction(self.direction)
        sign = sign_by_direction(self.direction)
        faces = []
        for left, right, bottom, top in self.rectangles:
            v1 = projection_inv((left, bottom), axis, self.offset)
            v2 = projection_inv((right + 1, bottom), axis, self.offset)
            v3 = projection_inv((right + 1, top + 1), axis, self.offset)
            v4 = projection_inv((left, top + 1), axis, self.offset)

            uv1 = (u_offset + left - self.aabb[0], v_offset + bottom - self.aabb[2])
            uv2 = (u_offset + right + 1 - self.aabb[0], v_offset + bottom - self.aabb[2])
            uv3 = (u_offset + right + 1 - self.aabb[0], v_offset + top + 1 - self.aabb[2])
            uv4 = (u_offset + left - self.aabb[0], v_offset + top + 1 - self.aabb[2])
            if self.transposed:
                uv2, uv4 = uv4, uv2
            if (sign > 0) != (axis == 1):
                faces.append(((v1, v2, v3, v4), (uv1, uv2, uv3, uv4), self.direction))
            else:
                faces.append(((v1, v4, v3, v2), (uv1, uv4, uv3, uv2), self.direction))
        return faces

    def set_uv(self, u_offset, v_offset, transposed):
        self.uv_offset = u_offset, v_offset
        self.transposed = transposed


def cut_model(model, direction, offset):
    axis = axis_by_direction(direction)
    sign = sign_by_direction(direction)
    size = model.get_size()

    empty_cut = np.zeros(projection(size, axis=axis), dtype=int)
    offset_back = offset + (-1 if sign > 0 else 0)
    offset_front = offset + (-1 if sign < 0 else 0)

    def is_valid(x):
        return 0 <= x < size[axis]

    slice_back = (empty_cut if not is_valid(offset_back) else get_slice(model.voxels_map, axis, offset_back))
    slice_front = (empty_cut if not is_valid(offset_front) else get_slice(model.voxels_map, axis, offset_front))

    cut = np.where(slice_front == 0, slice_back, 0)
    return [Part(np.where(c, cut, 0), aabb, direction, offset) for c, aabb in get_components(cut)]


'''
class TexturePiece:
    mod = 1000000007
    A = 1583

    def __init__(self, pid, data: np.ndarray):
        self.data = data
        self.shape = data.shape
        self.vertices = []
        self.pid = pid

    def get_hash(self):
        # hs = sum([(int(v) * pow(TexturePiece.A, i, TexturePiece.mod)) for i, v in enumerate(self.data.ravel())])
        # hs += self.shape[0] * TexturePiece.A * 331 + self.shape[1] * (TexturePiece.A - 1) * 997
        return self.pid


class Face:
    def __init__(self, vertices, texture_piece: TexturePiece, axis, direction):
        self.vertices = vertices
        self.texture_piece = texture_piece
        self.axis = axis
        self.direction = direction

    def __str__(self):
        vs = list(zip(self.vertices, self.texture_piece.vertices, ['', '', '', '']))
        if (self.direction > 0) != (self.axis == 1):
            vs = [vs[0], vs[3], vs[2], vs[1]]
        r = 'f ' + ' '.join(map(lambda v: '/'.join(map(str, v)), vs))
        return r


class Texture:
    def __init__(self, resolution=8, padding=6):
        self.pieces = {}
        self.resolution = resolution
        self.padding = padding
        self.texture_vertices = {}
        self.max_side = 0

    def get_piece(self, data):
        piece = TexturePiece(len(self.pieces), data)
        piece_hash = piece.get_hash()
        if piece_hash not in self.pieces:
            self.pieces[piece_hash] = piece
        return self.pieces[piece_hash]

    def get_piece_sides(self, piece):
        return (piece.shape[0] * self.resolution + self.padding * 2,
                piece.shape[1] * self.resolution + self.padding * 2)

    def draw_texture(self, palette):
        counter = 0
        texture_vertices = self.texture_vertices

        def get_index(x, y):
            nonlocal counter
            if (x, y) not in texture_vertices:
                counter += 1
                texture_vertices[(x, y)] = counter
            return texture_vertices[(x, y)]

        max_side = 0
        total_area = 0
        for piece in self.pieces.values():
            sides = self.get_piece_sides(piece)
            max_side = max(max_side, max(sides))
            total_area += sides[0] * sides[1]

        max_side = int(max(max_side, np.sqrt(total_area * 1.1)) * 0.8)
        v = 1
        while v < max_side:
            v *= 2
        max_side = v
        self.max_side = max_side
        colors = np.zeros((max_side, max_side), dtype=int)
        print('Texture side: ', max_side)
        for i, piece in enumerate(self.pieces.values()):
            ok = False
            print(i)
            for i in range(self.padding, colors.shape[0], 16):
                for j in range(self.padding, colors.shape[1], 16):
                    sides = self.get_piece_sides(piece)
                    if colors[i, j] > 0:
                        continue
                    if i + sides[0] >= colors.shape[0]:
                        continue
                    if j + sides[1] >= colors.shape[1]:
                        continue
                    if np.all(colors[i:i + sides[0], j:j + sides[1]] == 0):
                        v1 = (i + self.padding, j + self.padding)
                        v2 = (i - self.padding + sides[0], j + self.padding)
                        v3 = (i - self.padding + sides[0], j - self.padding + sides[1])
                        v4 = (i + self.padding, j - self.padding + sides[1])
                        vertices = [get_index(*v1), get_index(*v4), get_index(*v3), get_index(*v2)]
                        piece.vertices = vertices
                        colors[
                        i + self.padding: i - self.padding + sides[0],
                        j + self.padding: j - self.padding + sides[1]
                        ] = np.kron(piece.data, np.ones((self.resolution, self.resolution)))
                        ok = True
                    if ok:
                        break
                if ok:
                    break
            if not ok:
                print('blyat')

        print('Texture ready')
        q = Queue()
        for i in range(max_side):
            for j in range(max_side):
                if colors[i, j] != 0:
                    q.put((i, j))
        while not q.empty():
            i, j = q.get()
            for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                ni, nj = i + di, j + dj
                if ni < 0 or nj < 0 or ni >= max_side or nj >= max_side:
                    continue
                if colors[ni, nj] == 0:
                    colors[ni, nj] = colors[i, j]
                    q.put((ni, nj))

        print('Texture saving')

        lightpixels = np.zeros((max_side, max_side, 3), dtype=np.uint8)
        pixels = np.zeros((max_side, max_side, 4), dtype=np.uint8)
        pixels[:, :] = np.array([255, 0, 0, 255])

        for i in range(max_side):
            for j in range(max_side):
                if colors[i, j] > 0:
                    pixels[i, j] = np.array(palette.get_rgba(colors[i, j]))
                if colors[i, j] > 100:
                    lightpixels[i, j, :] = 255
        pixels = pixels[::, ::-1, ::].transpose(1, 0, 2)
        lightpixels = lightpixels[::, ::-1, ::].transpose(1, 0, 2)

        img = Image.fromarray(pixels, mode='RGBA')
        img.save("texture.png")
        img2 = Image.fromarray(lightpixels, mode='RGB')
        img2.save("texture_light.png")
        print('cc')

    def get_vertices(self):
        v = sorted(self.texture_vertices.keys(), key=lambda p: self.texture_vertices[p])
        return [[u[0] / self.max_side, u[1] / self.max_side] for u in v]
'''


class ObjModel:
    def __init__(self, vox_model: Model, palette, scale=0.05, texture_size=1024, emission_colors=()):
        vox_model_tr = vox_model.transposed()
        size = vox_model_tr.get_size()

        parts: List[Part] = []

        for direction in ALL_DIRECTIONS:
            axis = axis_by_direction(direction)
            for offset in range(size[axis] + 1):
                parts.extend(cut_model(vox_model_tr, direction, offset))

        def aabb_to_wh(aabb):
            return aabb[1] - aabb[0] + 1, aabb[3] - aabb[2] + 1

        rectangles = [p.get_data_inside_aabb() != 0 for p in parts]
        positions, side = pack(rectangles)

        color_map = np.zeros((side, side), dtype=int)

        for part, (i, j, tr) in zip(parts, positions):
            part.set_uv(i, j, tr)
            w, h = aabb_to_wh(part.aabb)
            if not tr:
                data = part.get_data_inside_aabb()
                color_map[i:i + w, j:j + h] = np.where(data != 0, data, color_map[i:i + w, j:j + h])
            else:
                data = part.get_data_inside_aabb().T
                color_map[i:i + h, j:j + w] = np.where(data != 0, data, color_map[i:i + h, j:j + w])

        voxel_texture_scale = int(np.floor(texture_size / side))

        color_map_expanded = np.kron(color_map, np.ones((voxel_texture_scale, voxel_texture_scale), dtype=int))

        color_map_final = np.zeros((texture_size, texture_size), dtype=int)

        color_map_final[:color_map_expanded.shape[0], :color_map_expanded.shape[1]] = color_map_expanded

        #color_map_final = fill_color_map(color_map_final)

        texture = np.zeros((*color_map_final.shape, 4), dtype=np.uint8)

        for i in range(texture.shape[0]):
            for j in range(texture.shape[1]):
                texture[i, j, :] = palette.get_rgba(color_map_final[i, j])

        texture = np.flip(texture.transpose((1, 0, 2)), axis=0)

        self.texture = texture

        self.translation = -vox_model_tr.origin
        self.scale = scale
        self.uv_scale = texture_size / voxel_texture_scale
        self.faces = []
        for part in parts:
            self.faces.extend(part.get_faces())

        if len(emission_colors) > 0:
            emission_texture = np.zeros((*color_map_final.shape, 3), dtype=np.uint8)
            for i in range(emission_texture.shape[0]):
                for j in range(emission_texture.shape[1]):
                    if color_map_final[i, j] in emission_colors:
                        emission_texture[i, j, :] = 255
            emission_texture = np.flip(emission_texture.transpose((1, 0, 2)), axis=0)

            self.emission_texture = emission_texture
        else:
            self.emission_texture = None

    def write_to_folder(self, folder_name):
        vertices_dict = {}
        uv_vertices_dict = {}

        def add(value, dictionary):
            if value not in dictionary:
                dictionary[value] = len(dictionary) + 1

        for vs, uvs, direction in self.faces:
            for v in vs:
                add(v, vertices_dict)
            for uv in uvs:
                add(uv, uv_vertices_dict)

        vertices = sorted(vertices_dict.keys(), key=lambda v_: vertices_dict[v_])
        uv_vertices = sorted(uv_vertices_dict.keys(), key=lambda v_: uv_vertices_dict[v_])

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        def transform_v(v_):
            return tuple((self.translation + np.array(v_)) * self.scale)

        def transform_uv(uv_):
            return tuple(np.array(uv_) / self.uv_scale)

        obj_file = open(os.path.join(folder_name, 'object.obj'), 'w')

        obj_file.write('# VoxExporter (https://github.com/AndrewB330/VoxExporter)\n\n')

        obj_file.write(f'# {len(vertices)} vertices\n')
        obj_file.write(f'# {len(uv_vertices)} uv vertices\n')
        obj_file.write(f'# {len(self.faces)} faces\n\n')

        obj_file.write('\n'.join(map(lambda d_: 'vn %d %d %d' % tuple(normal_by_direction(d_)), ALL_DIRECTIONS)))
        obj_file.write('\n\n')
        obj_file.write('\n'.join(map(lambda v_: 'v %f %f %f' % transform_v(v_), vertices)))
        obj_file.write('\n\n')
        obj_file.write('\n'.join(map(lambda uv_: 'vt %f %f' % transform_uv(uv_), uv_vertices)))
        obj_file.write('\n\n')

        def face_to_str(face):
            vs_, uvs_, direction_ = face
            vs_ = map(lambda v_: vertices_dict[v_], vs_)
            uvs_ = map(lambda uv_: uv_vertices_dict[uv_], uvs_)
            return 'f ' + ' '.join(map(lambda a_: '/'.join(map(str, a_)), zip(vs_, uvs_, [direction_.value + 1] * 4)))

        obj_file.write('\n'.join(map(face_to_str, self.faces)))

        obj_file.close()

        Image.fromarray(self.texture).save(os.path.join(folder_name, 'texture.png'))
        if self.emission_texture is not None:
            Image.fromarray(self.emission_texture).save(os.path.join(folder_name, 'emission.png'))
