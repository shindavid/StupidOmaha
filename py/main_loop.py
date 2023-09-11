#!/usr/bin/env python3

import argparse
import os

from manager import Manager


class Args:
    tag: str

    @staticmethod
    def load(args):
        Args.tag = args.tag
        assert Args.tag, 'Required option: --tag/-t'


def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--tag', help='tag for this run (e.g. "v1")')

    args = parser.parse_args()
    Args.load(args)


def main():
    load_args()
    base_dir = os.path.join('/media/dshin/stupidomaha', Args.tag)
    manager = Manager(base_dir)
    manager.makedirs()
    manager.run()


if __name__ == '__main__':
    main()
