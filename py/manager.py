from collections import defaultdict
from natsort import natsorted
import os
import shutil
import sys
import tempfile
import time
import torch
from torch.optim import Optimizer, SGD, Adam
from torch.utils.data import DataLoader
from typing import Callable, List, Optional, Tuple

from py_util import make_hidden_filename, timed_print
from neural_net import LearningTarget, NeuralNet
import subprocess_util
from tensor_dataset import TensorDataset


Generation = int

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


RANKS = '23456789TJQKA'
SUITS = 'cdhs'
NUM_CARDS = 52
NUM_PLAYERS = 6


def card_repr(k):
    return RANKS[k % 13] + SUITS[k // 13]


def input_repr(row):
    card_indices = [int(x) for x in torch.where(row[:NUM_CARDS])[0]]
    assert len(card_indices) == 4
    card_indices.sort(key=lambda c: (-(c % 13), c // 13))
    cards_repr = ''.join(card_repr(c) for c in card_indices)
    pos_indices = row[NUM_CARDS: NUM_CARDS + NUM_PLAYERS - 1]
    call_indices = row[NUM_CARDS + NUM_PLAYERS - 1:]

    action_str = ['_' for _ in range(NUM_PLAYERS)]
    action_str[-1] = 'B'
    for k in range(NUM_PLAYERS - 1):
        if pos_indices[k]:
            break
        action_str[k] = 'C' if call_indices[k] else 'F'
    return cards_repr + ' ' + ''.join(action_str)


def apply_mask(tensor: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return tensor
    return tensor[mask]


class EvaluationResults:
    def __init__(self, labels, outputs, loss):
        self.labels = labels
        self.outputs = outputs
        self.loss = loss

    def __len__(self):
        return len(self.labels)


class TrainingSubStats:
    max_descr_len = 0

    def __init__(self, target: LearningTarget):
        self.target = target
        self.accuracy_num = 0.0
        self.loss_num = 0.0
        self.den = 0

        TrainingSubStats.max_descr_len = max(TrainingSubStats.max_descr_len, len(self.descr))

    @property
    def descr(self) -> str:
        return self.target.name

    def update(self, results: EvaluationResults):
        n = len(results)
        self.accuracy_num += self.target.get_num_correct_predictions(results.outputs, results.labels)
        self.loss_num += float(results.loss.item()) * n
        self.den += n

    def accuracy(self):
        return self.accuracy_num / self.den if self.den else 0.0

    def loss(self):
        return self.loss_num / self.den if self.den else 0.0

    def dump(self):
        tuples = [
            (' accuracy:', self.accuracy()),
            (' loss:', self.loss()),
        ]
        max_str_len = max([len(t[0]) for t in tuples]) + TrainingSubStats.max_descr_len
        for key, value in tuples:
            full_key = self.descr + key
            print(f'{full_key.ljust(max_str_len)} %8.6f' % value)


class TrainingStats:
    def __init__(self, net: NeuralNet):
        self.substats_list = [TrainingSubStats(target) for target in net.learning_targets]

    def update(self, results_list: List[EvaluationResults]):
        for results, substats in zip(results_list, self.substats_list):
            substats.update(results)

    def dump(self):
        for substats in self.substats_list:
            substats.dump()


class ModelingArgs:
    window_c = 250000
    window_alpha = 0.75
    window_beta = 0.4
    minibatch_size = 256
    learning_rate = 6e-5
    momentum = 0.9
    weight_decay = 6e-5
    snapshot_steps = 2048


def compute_n_window(n_total: int) -> int:
    """
    From Appendix C of KataGo paper.

    https://arxiv.org/pdf/1902.10565.pdf
    """
    c = ModelingArgs.window_c
    alpha = ModelingArgs.window_alpha
    beta = ModelingArgs.window_beta
    return min(n_total, int(c * (1 + beta * ((n_total / c) ** alpha - 1) / alpha)))


class PathInfo:
    def __init__(self, path: str):
        self.path: str = path
        self.generation: Generation = -1

        payload = os.path.split(path)[1].split('.')[0]
        tokens = payload.split('-')
        for t, token in enumerate(tokens):
            if token == 'gen':
                self.generation = int(tokens[t+1])


class SelfPlayResults:
    def __init__(self, stdout: str):
        self.mappings = {}
        for line in stdout.splitlines():
            if not line.startswith('RESULT '):
                continue
            # RESULT abc def: 1234
            tokens = line.split(':')
            assert len(tokens) == 2, f'Unexpected line: {line}'
            key = tokens[0][tokens[0].find(' ') + 1:]
            value = tokens[1].strip()
            self.mappings[key] = value


class SelfPlayProcData:
    def __init__(self, cmd: List[str], gen: Generation, games_dir: str):
        if os.path.exists(games_dir):
            # This likely means that we are resuming a previous run that already wrote some games to
            # this directory. In principle, we could make use of those games. However, that
            # complicates the tracking of some stats, like the total amount of time spent on
            # self-play. Since not a lot of compute/time is spent on each generation, we just blow
            # away the directory to make our lives simpler
            shutil.rmtree(games_dir)

        self.proc_complete = False
        self.proc = subprocess_util.Popen(cmd)
        self.gen = gen
        self.games_dir = games_dir
        timed_print(f'Running gen-{gen} self-play [{self.proc.pid}]: {" ".join(cmd)}')

        if gen == 0:
            results = self.wait_for_completion()
            SelfPlayProcData.print_results(results)

    def terminate(self, timeout: Optional[int] = None, finalize_games_dir=True,
                  expected_return_code: Optional[int] = 0, print_results=True):
        if self.proc_complete:
            return
        kill_file = os.path.join(self.games_dir, 'kill.txt')
        os.system(f'touch {kill_file}')  # signals c++ process to stop
        results = self.wait_for_completion(
            timeout=timeout, finalize_games_dir=finalize_games_dir,
            expected_return_code=expected_return_code)
        if print_results:
            SelfPlayProcData.print_results(results)

    def wait_for_completion(self, timeout: Optional[int] = None, finalize_games_dir=True,
                            expected_return_code: Optional[int] = 0) -> SelfPlayResults:
        timed_print(f'Waiting for self-play proc [{self.proc.pid}] to complete...')
        stdout = subprocess_util.wait_for(self.proc, timeout=timeout, expected_return_code=expected_return_code)
        results = SelfPlayResults(stdout)
        if finalize_games_dir:
            Manager.finalize_games_dir(self.games_dir, results)
        timed_print(f'Completed gen-{self.gen} self-play [{self.proc.pid}]')
        self.proc_complete = True
        return results

    @staticmethod
    def print_results(results: SelfPlayResults):
        call_counts = defaultdict(lambda: defaultdict(int))
        total_counts = defaultdict(lambda: defaultdict(int))
        for key, value in results.mappings.items():
            tokens = key.split('-')
            if tokens[0] not in ('call', 'fold'):
                continue
            seat = int(tokens[1])
            num_prev_calls = int(tokens[2])
            if num_prev_calls > 1:
                continue
            opening = 1 if (num_prev_calls == 0) else 0
            total_counts[seat][opening] += int(value)
            if tokens[0] == 'call':
                call_counts[seat][opening] += int(value)

        SelfPlayProcData.print_header()
        for seat in range(5):
            open_num = call_counts[seat][1]
            open_den = total_counts[seat][1]
            post_num = call_counts[seat][0]
            post_den = total_counts[seat][0]

            open_pct = open_num / open_den if open_den else None
            post_pct = post_num / post_den if post_den else None
            cols = [
                str(seat),
                '%2.f%%' % (100 * open_pct) if open_pct is not None else '',
                '%2.f%%' % (100 * post_pct) if post_pct is not None else '',
                '%6d' % open_den if open_den else '',
                '%6d' % post_den if post_den else '',
                ]
            for col in cols:
                print(f"{col:>{SelfPlayProcData.column_width()}}", end="")
            print()  # Newline at the end

    @staticmethod
    def column_width():
        return 12

    @staticmethod
    def print_header():
        super_columns = ["%", "N"]
        regular_columns = ["Seat", "Open", "After1Call", "Open", "After1Call"]

        # Calculate spacing for super columns
        spacing_per_regular_col = SelfPlayProcData.column_width()
        super_col0_width = 2 * spacing_per_regular_col - len(regular_columns[1])
        super_col3_width = spacing_per_regular_col + len(regular_columns[3])
        super_col2_width = spacing_per_regular_col - len(regular_columns[3])
        super_col1_width = 5 * spacing_per_regular_col - super_col0_width - super_col2_width - super_col3_width

        # Print super columns
        print("".center(super_col0_width), end="")
        print(super_columns[0].center(super_col1_width), end="")
        print("".center(super_col2_width), end="")
        print(super_columns[1].center(super_col3_width), end="")
        print()  # Newline at the end

        # Print regular columns
        for col in regular_columns:
            print(f"{col:>{spacing_per_regular_col}}", end="")
        print()  # Newline at the end


class Manager:
    def __init__(self, base_dir: str):
        self.net = None
        self.opt = None
        self.self_play_proc_data = None
        self.py_cuda_device = 1  # TODO: configure this

        self.base_dir = base_dir
        self.stdout_filename = os.path.join(self.base_dir, 'stdout.txt')
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.checkpoints_dir = os.path.join(self.base_dir, 'checkpoints')
        self.self_play_data_dir = os.path.join(self.base_dir, 'self-play-data')

    @property
    def py_cuda_device_str(self) -> str:
        return f'cuda:{self.py_cuda_device}'

    def makedirs(self):
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.self_play_data_dir, exist_ok=True)

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

    def get_latest_checkpoint_info(self) -> Optional[PathInfo]:
        return Manager.get_latest_info(self.checkpoints_dir)

    def get_checkpoint_filename(self, gen: Generation) -> str:
        return os.path.join(self.checkpoints_dir, f'gen-{gen}.ptc')

    def get_latest_model_generation(self):
        info = Manager.get_latest_info(self.models_dir)
        return 0 if info is None else info.generation

    def get_self_play_data_subdir(self, gen: Generation) -> str:
        return os.path.join(self.self_play_data_dir, f'gen-{gen}')

    def get_model_filename(self, gen: Generation) -> str:
        return os.path.join(self.models_dir, f'gen-{gen}.ptj')

    def get_self_play_proc(self):
        gen = self.get_latest_model_generation()

        games_dir = self.get_self_play_data_subdir(gen)
        bin_path = os.path.join(REPO_ROOT, f'build/Release/bin/sim')

        n_hands = 10000 if gen == 0 else 0
        cmd = [bin_path, '-h', str(n_hands), '-o', games_dir]
        if gen:
            model = self.get_model_filename(gen)
            kill_file = os.path.join(games_dir, 'kill.txt')
            cmd.extend([
                '-f', model,
                '-k', kill_file
            ])

        return SelfPlayProcData(cmd, gen, games_dir)

    @staticmethod
    def finalize_games_dir(games_dir: str, results: SelfPlayResults):
        timed_print('Finalizing games dir: %s' % games_dir)
        n_training_rows = int(results.mappings['n_training_rows'])

        done_file = os.path.join(games_dir, 'done.txt')
        tmp_done_file = make_hidden_filename(done_file)
        with open(tmp_done_file, 'w') as f:
            f.write(f'n_training_rows={n_training_rows}\n')
        os.rename(tmp_done_file, done_file)

    def init_logging(self, filename: str):
        self.log_file = open(filename, 'a')
        sys.stdout = self
        sys.stderr = self

    def write(self, msg):
        sys.__stdout__.write(msg)
        if self.log_file is not None:
            self.log_file.write(msg)
        self.flush()

    def flush(self):
        sys.__stdout__.flush()
        if self.log_file is not None:
            self.log_file.flush()

    def run(self):
        self.init_logging(self.stdout_filename)
        while True:
            self.self_play_proc_data = self.get_self_play_proc()
            self.train_step(pre_commit_func=lambda: self.self_play_proc_data.terminate(timeout=300))

    def get_net_and_optimizer(self, loader: 'DataLoader') -> Tuple[NeuralNet, Optimizer]:
        if self.net is not None:
            return self.net, self.opt

        checkpoint_info = self.get_latest_checkpoint_info()
        if checkpoint_info is None:
            self.net = NeuralNet()
            timed_print(f'Creating new net')
        else:
            gen = checkpoint_info.generation
            checkpoint_filename = self.get_checkpoint_filename(gen)
            timed_print(f'Loading checkpoint: {checkpoint_filename}')

            # copying the checkpoint to somewhere local first seems to bypass some sort of filesystem issue
            with tempfile.TemporaryDirectory() as tmp:
                tmp_checkpoint_filename = os.path.join(tmp, 'checkpoint.ptc')
                shutil.copy(checkpoint_filename, tmp_checkpoint_filename)
                self.net = NeuralNet.load_checkpoint(tmp_checkpoint_filename)

        self.net.cuda(device=1)
        self.net.train()

        learning_rate = ModelingArgs.learning_rate
        momentum = ModelingArgs.momentum
        weight_decay = ModelingArgs.weight_decay
        self.opt = Adam(self.net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # self.opt = SGD(self.net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

        # TODO: SWA, cyclic learning rate

        return self.net, self.opt

    def train_step(self, pre_commit_func: Optional[Callable[[], None]] = None):
        """
        Performs a train-step. This performs N minibatch-updates of size S, where:

        N = ModelingArgs.snapshot_steps
        S = ModelingArgs.minibatch_size

        After the last minibatch update is complete, but before the model is committed to disk, pre_commit_func() is
        called. We use this function shutdown the c++ self-play process. This is necessary because the self-play process
        prints important metadata to stdout, and we don't want to commit a model for which we don't have the metadata.
        """
        print('******************************')
        gen = self.get_latest_model_generation() + 1
        timed_print(f'Train gen:{gen}')

        for_loop_time = 0
        t0 = time.time()
        steps = 0
        while True:
            dataset = TensorDataset(self.self_play_data_dir)
            n_total_positions = dataset.size
            dataset.resize(compute_n_window(dataset.size))
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=ModelingArgs.minibatch_size,
                num_workers=4,
                pin_memory=True,
                shuffle=True)

            net, optimizer = self.get_net_and_optimizer(loader)
            dataset.set_key_order(net.target_names())

            loss_fns = [target.loss_fn() for target in net.learning_targets]

            suffix = ''
            if steps:
                suffix = f' (minibatches processed: {steps})'
            timed_print(f'Sampling from the {len(dataset)} most recent positions among '
                        f'{n_total_positions} total positions{suffix}')

            stats = TrainingStats(net)
            print_count = 5  # set to positive number to see print of example hands
            for data in loader:
                t1 = time.time()
                inputs = data[0]
                labels_list = data[1:]
                inputs = inputs.type(torch.float32).to(self.py_cuda_device_str)

                labels_list = [labels.to(self.py_cuda_device_str) for labels in labels_list]

                optimizer.zero_grad()
                outputs_list = net(inputs)
                assert len(outputs_list) == len(labels_list)

                labels_list = [labels.reshape((labels.shape[0], -1)) for labels in labels_list]
                outputs_list = [outputs.reshape((outputs.shape[0], -1)) for outputs in outputs_list]

                masks = [target.get_mask(labels) for labels, target in zip(labels_list, net.learning_targets)]

                labels_list = [apply_mask(labels, mask) for mask, labels in zip(masks, labels_list)]
                outputs_list = [apply_mask(outputs, mask) for mask, outputs in zip(masks, outputs_list)]

                loss_list = [loss_fn(outputs, labels) for loss_fn, outputs, labels in
                             zip(loss_fns, outputs_list, labels_list)]

                loss = sum([loss * target.loss_weight for loss, target in zip(loss_list, net.learning_targets)])

                results_list = [EvaluationResults(labels, outputs, loss) for labels, outputs, loss in
                                zip(labels_list, outputs_list, loss_list)]

                stats.update(results_list)

                if print_count:
                    i = inputs[0]
                    o = labels_list[0][0]
                    p = outputs_list[0][0]
                    print('%s %+6.3f %+6.3f' % (input_repr(i), o, p))
                    print_count -= 1

                loss.backward()
                optimizer.step()
                steps += 1
                t2 = time.time()
                for_loop_time += t2 - t1
                if steps == ModelingArgs.snapshot_steps:
                    break
            stats.dump()
            if steps >= ModelingArgs.snapshot_steps:
                break

        t3 = time.time()
        total_time = t3 - t0
        data_loading_time = total_time - for_loop_time

        timed_print(f'Gen {gen} training complete ({steps} minibatch updates)')
        timed_print(f'Data loading time: {data_loading_time:10.3f} seconds')
        timed_print(f'Training time:     {for_loop_time:10.3f} seconds')

        if pre_commit_func:
            pre_commit_func()

        checkpoint_filename = self.get_checkpoint_filename(gen)
        model_filename = self.get_model_filename(gen)
        tmp_checkpoint_filename = make_hidden_filename(checkpoint_filename)
        tmp_model_filename = make_hidden_filename(model_filename)
        net.save_checkpoint(tmp_checkpoint_filename)
        net.save_model(tmp_model_filename)
        os.rename(tmp_checkpoint_filename, checkpoint_filename)
        os.rename(tmp_model_filename, model_filename)
        timed_print(f'Checkpoint saved: {checkpoint_filename}')
        timed_print(f'Model saved: {model_filename}')
