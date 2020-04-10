from typing import List

from bytes_reader import BytesReader
from model import Group, Transform, Shape, Model
from palette import Palette

SUPPORTED_VERSIONS = [150]


class Chunk:
    def __init__(self, chunk_type: str, size_bytes: int, content: bytes, children: List['Chunk']):
        self.chunk_type = chunk_type
        self.size_bytes = size_bytes
        self.content = content
        self.children = children

    def __str__(self, prefix=''):
        res = prefix + f'[{self.chunk_type}]'
        if len(self.children) == 0:
            return res
        return res + '\n' + '\n'.join(c.__str__(prefix + '  ') for c in self.children)

    def get_children(self, children_type):
        return [c for c in self.children if c.chunk_type == children_type]

    def get_child(self, children_type):
        all_ = [c for c in self.children if c.chunk_type == children_type]
        return None if len(all_) == 0 else all_[0]


def decode_chunks(data: bytes):
    reader = BytesReader(data)

    chunk_type = reader.read(4).decode()

    content_size, children_content_size = reader.read_int(), reader.read_int()
    size = 4 + 4 + 4 + content_size + children_content_size

    content = reader.read(content_size)
    children_content = reader.read(children_content_size)
    children = []

    while len(children_content) > 0:
        child = decode_chunks(children_content)
        children_content = children_content[child.size_bytes:]
        children.append(child)

    return Chunk(chunk_type, size, content, children)


def build_graph(groups: List[Group], transforms: List[Transform], shapes: List[Shape], models: List[Model]):
    id_to_node = {node.node_id: node for node in (transforms + groups + shapes)}
    for transform in transforms:
        transform.child = id_to_node[transform.child_id]
    for group in groups:
        group.children = [id_to_node[cid] for cid in group.children_ids]
    for shape in shapes:
        shape.model = models[shape.model_id]
    root = transforms[0]  # not always, but ox for current version
    return root


def read(filename):
    reader = BytesReader(open(filename, 'rb').read())
    magic = reader.read(4)
    if magic != b'VOX ':
        print('Given file is not in VOX file-format')
        return None

    version = int.from_bytes(reader.read(4), byteorder='little')
    assert (version in SUPPORTED_VERSIONS)

    main_chunk = decode_chunks(reader.read())
    assert (main_chunk.chunk_type == 'MAIN')

    size = main_chunk.get_children('SIZE')
    xyzc = main_chunk.get_children('XYZI')
    ngrp = main_chunk.get_children('nGRP')
    ntrn = main_chunk.get_children('nTRN')
    nshp = main_chunk.get_children('nSHP')
    rgba = main_chunk.get_child('RGBA')
    assert (len(size) == len(xyzc))

    models = [Model.from_bytes(s.content, x.content) for s, x in zip(size, xyzc)]
    palette = Palette.from_bytes(rgba.content) if rgba is not None else Palette.default()

    groups = [Group.from_bytes(c.content) for c in ngrp]
    transforms = [Transform.from_bytes(c.content) for c in ntrn]
    shapes = [Shape.from_bytes(c.content) for c in nshp]

    root = build_graph(groups, transforms, shapes, models)

    return root, palette
