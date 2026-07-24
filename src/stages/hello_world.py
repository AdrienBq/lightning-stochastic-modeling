from fire import Fire

from __init__ import root_path
# === other src imports go below this line ===


def hello_world():
    print('hello world!')


if __name__ == '__main__':
    Fire(hello_world)