import os


def rand_id() -> str:
    return os.urandom(4).hex()
