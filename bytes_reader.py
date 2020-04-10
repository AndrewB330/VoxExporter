from typing import Tuple


class BytesReader:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def read(self, num: int = -1) -> bytes:
        if num < 0:
            res = self.data[self.offset:]
            self.offset = len(self.data)
            return res
        res = self.data[self.offset: self.offset + num]
        self.offset += num
        return res

    def read_int(self, signed=False):
        return int.from_bytes(self.read(4), byteorder='little', signed=signed)

    def read_ints(self, n: int = 1, signed=False) -> Tuple[int, ...]:
        return tuple(self.read_int(signed) for _ in range(n))

    def read_byte(self) -> int:
        return int.from_bytes(self.read(1), byteorder='little')

    def read_bytes(self, n: int = 1) -> Tuple[int, ...]:
        return tuple(self.read_byte() for _ in range(n))

    def read_string(self) -> str:
        length = self.read_int()
        return self.read(length).decode()

    def read_dict(self) -> dict:
        d = {}
        for _ in range(self.read_int()):
            key, value = self.read_string(), self.read_string()
            d[key] = value
        return d

    def is_eof(self) -> bool:
        return self.offset >= len(self.data)
