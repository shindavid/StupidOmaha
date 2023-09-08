from natsort import natsorted
import os
from typing import List, Optional


Generation = int

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class PathInfo:
    def __init__(self, path: str):
        self.path: str = path
        self.generation: Generation = -1

        payload = os.path.split(path)[1].split('.')[0]
        tokens = payload.split('-')
        for t, token in enumerate(tokens):
            if token == 'gen':
                self.generation = int(tokens[t+1])


class Manager:
    def run(self, base_dir: str):
        self.net = None
        self.opt = None
        self.self_play_proc_data = None
        self.py_cuda_device = 1  # TODO: configure this

        self.base_dir = base_dir
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.train_data_dir = os.path.join(self.base_dir, 'train_data')
        self.checkpoints_dir = os.path.join(self.base_dir, 'checkpoints')

    def makedirs(self):
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.train_data_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    @staticmethod
    def get_ordered_subpaths(path: str) -> List[str]:
        subpaths = list(natsorted(f for f in os.listdir(path)))
        return [f for f in subpaths if not f.startswith('.')]

    @staticmethod
    def get_latest_info(path: str) -> Optional[PathInfo]:
        subpaths = Manager.get_ordered_subpaths(path)
        if not subpaths:
            return None
        return PathInfo(subpaths[-1])

    def get_latest_model_generation(self):
        info = Manager.get_latest_info(self.models_dir)
        return 0 if info is None else info.generation

    def get_self_play_data_subdir(self, gen: Generation) -> str:
        return os.path.join(self.self_play_data_dir, f'gen-{gen}')

    def get_self_play_proc(self):
        gen = self.get_latest_model_generation()

        games_dir = self.get_self_play_data_subdir(gen)
        bin_path = os.path.join(REPO_ROOT, f'build/bin/sim')

        if gen == 0:
            n_hands = 10000
        else:
            n_hands = 0

        args = ['-h', n_hands, '-o', games_dir]
        base_player_args = []  # '--no-forced-playouts']
        if gen:
            model = self.get_model_filename(gen)
            base_player_args.extend(['-m', model])

        player_args = [
            '--type=MCTS-T',
            '--name=MCTS',
            '-g', games_dir,
        ] + base_player_args

        player2_args = [
            '--name=MCTS2',
            '--copy-from=MCTS',
        ]

        self_play_cmd = [
            bin_tgt,
            '-G', n_games,
            '--player', '"%s"' % (' '.join(map(str, player_args))),
            '--player', '"%s"' % (' '.join(map(str, player2_args))),
        ]

        if n_games == 0:
            kill_file = os.path.join(games_dir, 'kill.txt')
            self_play_cmd.extend([
                '--kill-file', kill_file
            ])

        competitive_player_args = ['--type=MCTS-C'] + base_player_args
        competitive_player_str = '%s --player "%s"\n' % (
            bin_tgt, ' '.join(map(str, competitive_player_args)))
        player_filename = os.path.join(self.players_dir, f'gen-{gen}.txt')
        with open(player_filename, 'w') as f:
            f.write(competitive_player_str)

        self_play_cmd = ' '.join(map(str, self_play_cmd))
        return SelfPlayProcData(self_play_cmd, n_games, gen, games_dir)

    def run(self):
        while True:
            self.self_play_proc_data = self.get_self_play_proc()
