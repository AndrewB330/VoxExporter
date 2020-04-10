from queue import Queue

import numpy as np

STEPS_2D = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def decode_rotation(encoded: int):
    """
    Decodes orthogonal rotation matrix from integer representation described in:
    https://github.com/ephtracy/voxel-model/blob/master/MagicaVoxel-file-format-vox-extension.txt

    1) first two bits determine index of non-zero column in first row
    2) next two bits determine index of non-zero column in second row
    3) index of non-zero column in third row can be determined after knowing two previous
    4) next three bits determine sign of one in each row (non-zero bit means -1, otherwise +1)

    Example:
     0  1  0
     0  0 -1
    -1  0  0
    0b1101001 = 105 | 0b[110][10][01] (bits are read from left to right)

    :param encoded: integer representation of rotation matrix
    :return: 3-dimensional rotation matrix (numpy array with shape (3, 3))
    """
    if encoded == 0:
        return np.identity(3)
    first_col, second_col = (encoded & 0b11), (encoded & 0b1100) >> 2
    signs = (encoded & 0b10000) >> 4, (encoded & 0b100000) >> 5, (encoded & 0b1000000) >> 6
    cols = [first_col, second_col, [i for i in range(3) if first_col != i and second_col != i][0]]
    mat = np.zeros((3, 3))
    for i in range(3):
        mat[i, cols[i]] = -1 if signs[i] else 1
    return mat


def decode_translation(string: str):
    """
    Decodes translation vector from string
    String has the following format: "%d %d %d"
    Example: "-5 11 0"
    :param string: string representation of translation vector
    :return: 3-dimensional translation vector (numpy array with shape (3))
    """
    x, y, z = map(int, string.split())
    return np.array([x, y, z])


def projection(pos, axis):
    if type(pos) == np.ndarray:
        return np.hstack([pos[:axis], pos[axis + 1:]])
    return pos[:axis] + pos[axis + 1:]


def projection_inv(pos, axis, at):
    if type(pos) == np.ndarray:
        return np.hstack([pos[:axis], np.array([at]), pos[axis:]])
    if type(pos) == tuple:
        return pos[:axis] + (at,) + pos[axis:]
    return pos[:axis] + [at] + pos[axis:]


def get_slice(arr, axis, at):
    if axis == 0:
        return arr[at, :, :]
    if axis == 1:
        return arr[:, at, :]
    return arr[:, :, at]


def find_position(table, w, h, padding, start_row=0):
    for i in range(max(start_row, padding), table.shape[0] - w - padding + 1):
        for j in range(padding, table.shape[1] - h - padding + 1):
            if np.sum(table[i - padding:i + w + padding, j - padding:j + h + padding]) == 0:
                return i, j
    return -1, -1


def pack(masks, padding=1):
    n = len(masks)
    # area = sum(map(lambda m: (m.shape[0] + padding) * (m.shape[1] + padding), masks))
    area = sum(map(lambda m: (m.shape[0] + m.shape[1] + padding) * padding + np.sum(m), masks))
    min_side = max(map(lambda m: max(m.shape), masks)) + padding * 2

    current_side = max(int(np.ceil(np.sqrt(area))), min_side)

    order = sorted(list(range(n)), key=lambda i: -masks[i].shape[0] * masks[i].shape[1])

    def try_pack(side):
        table = np.zeros((side, side), dtype=int)
        result = [(0, 0, 0)] * n
        ema_i = 0.0
        for step, index in enumerate(order):
            w, h = masks[index].shape
            i, j = find_position(table, w, h, padding, int(ema_i))
            transposed = False

            if i < 0 or j < 0:
                i, j = find_position(table, h, w, padding, int(ema_i))
                transposed = True

            if i < 0 or j < 0:
                return None

            ema_i = (ema_i * 0.3 + i * 0.7)
            if step % 30 == 0:
                ema_i = 0
            result[index] = (i, j, transposed)
            if not transposed:
                table[i:i + w, j:j + h] = masks[index]
            else:
                table[i:i + h, j:j + w] = masks[index].T
        print(np.sum(table) * 1.0 / table.shape[0] / table.shape[1])
        return result

    print('packing...')
    pack_result = try_pack(current_side)
    print(current_side)
    while pack_result is None:
        current_side = (current_side * 105) // 100 + 1
        print(current_side)
        pack_result = try_pack(current_side)
    return pack_result, current_side


def fill_color_map(color_map):
    q = Queue()
    for i in range(color_map.shape[0]):
        for j in range(color_map.shape[1]):
            if color_map[i, j] != 0:
                q.put((i, j))
    while not q.empty():
        ci, cj = q.get()
        for di, dj in STEPS_2D:
            ni, nj = ci + di, cj + dj
            if ni < 0 or nj < 0 or ni >= color_map.shape[0] or nj >= color_map.shape[1]:
                continue
            if color_map[ni, nj] == 0:
                color_map[ni, nj] = color_map[ci, cj]
                q.put((ni, nj))
    return color_map
