from model import merge_models
from obj_exporter import ObjModel
from vox_reader import read


def main():
    name = 'ceil_pipes'
    root, palette = read(f'vox/{name}.vox')
    merged = merge_models(root)
    obj = ObjModel(merged, palette, texture_size=2048)
    obj.write_to_folder(f'exported/{name}')


if __name__ == '__main__':
    main()
