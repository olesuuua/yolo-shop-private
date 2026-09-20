"""Bounded length-prefixed messages over a private parent/worker socket."""

import struct

MAX_MESSAGE_BYTES = 8_000_000


def _read_exact(connection, size):
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise EOFError("Paddle worker connection closed.")
        data.extend(chunk)
    return bytes(data)


def receive(connection):
    size, = struct.unpack("!I", _read_exact(connection, 4))
    if not 0 < size <= MAX_MESSAGE_BYTES:
        raise ValueError("Invalid Paddle worker message size.")
    return _read_exact(connection, size)


def send(connection, data):
    if not 0 < len(data) <= MAX_MESSAGE_BYTES:
        raise ValueError("Invalid Paddle worker message size.")
    connection.sendall(struct.pack("!I", len(data)) + data)
