"""
Wrapper around torch's nn.Module that facilitates caching and save/load mechanics.
"""
import abc
import copy
import os
from typing import Dict, List, Optional

import torch
from torch import nn as nn
from torch.nn import functional as F


ENABLE_CUDA = True
NUM_CARDS = 52
MAX_NUM_PLAYERS = 6
INPUT_SIZE = NUM_CARDS + 2 * (MAX_NUM_PLAYERS - 1)


class LearningTarget:
    """
    A LearningTarget corresponds to an output head of a NeuralNet. It bundles a number of useful pieces of information
    together:

    - The name of the head, to facilitate matching up with c++ generated data
    - The loss function to use for this head
    - The weight to assign to the loss produced by this head
    - Whether or how to mask rows of data
    - An accuracy measurement function
    """

    def __init__(self, name: str, loss_weight: float):
        self.name = name
        self.loss_weight = loss_weight

    @abc.abstractmethod
    def loss_fn(self) -> nn.Module:
        pass

    def get_mask(self, labels: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Assumes labels is a 2D tensor of shape (batch_size, num_labels). Returns a 1D tensor of shape (batch_size,)

        If no mask should be applied, return None. This is the default behavior; derived classes can override this.
        """
        return None

    @abc.abstractmethod
    def get_num_correct_predictions(self, outputs: torch.Tensor, labels: torch.Tensor) -> int:
        pass


class ValueHead(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256):
        super(ValueHead, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class ValueTarget(LearningTarget):
    def __init__(self, name: str, loss_weight: float, correct_threshold: float):
        super(ValueTarget, self).__init__(name, loss_weight)
        self.correct_threshold = correct_threshold

    def loss_fn(self) -> nn.Module:
        return nn.MSELoss()

    def get_num_correct_predictions(self, outputs: torch.Tensor, labels: torch.Tensor) -> int:
        deltas = abs(outputs - labels)
        return int(sum(deltas < self.correct_threshold))


class NeuralNet(nn.Module):
    def __init__(self, n_nodes_per_layer: int = 256, n_layers: int = 5):
        super(NeuralNet, self).__init__()
        self.first_layer = nn.Linear(INPUT_SIZE, n_nodes_per_layer)
        self.layers = nn.ModuleList(
            [nn.Linear(n_nodes_per_layer, n_nodes_per_layer)
             for _ in range(n_layers - 1)]
        )

        # heads and learning_targets are parallel lists, with the same length
        self.heads = nn.ModuleList()
        self.learning_targets: List[LearningTarget] = []

        self.add_head(ValueHead(n_nodes_per_layer), ValueTarget('output', 1.0, 0.05))

    def forward(self, x):
        x = self.first_layer(x)
        x = F.relu(x)
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
        return tuple(head(x) for head in self.heads)

    @classmethod
    def load_model(cls, filename: str, verbose: bool = False, eval_mode: bool = True) -> torch.jit.ScriptModule:
        """
        Loads a model previously saved to disk via save(). This uses torch.jit.load(), which returns a
        torch.jit.ScriptModule, which looks/feels/sounds like nn.Module, but is not exactly the same thing.

        For convenience, as a side-effect, this calls torch.set_grad_enabled(False), which mutates torch's global
        state.
        """
        if verbose:
            print(f'Loading model from {filename}')
        net = torch.jit.load(filename)
        if ENABLE_CUDA:
            net.to('cuda')
        if eval_mode:
            torch.set_grad_enabled(False)
            net.eval()
        else:
            net.train()
        if verbose:
            print(f'Model successfully loaded!')
        return net

    def save_model(self, filename: str, verbose: bool = False):
        """
        Saves this network to disk, from which it can be loaded either by c++ or by python. Uses the
        torch.jit.trace() function to accomplish this.

        Note that prior to saving, we "freeze" the model, by switching it to eval mode and disabling gradient.
        The documentation seems to imply that this is an important step:

        "...In the returned :class:`ScriptModule`, operations that have different behaviors in ``training`` and
         ``eval`` modes will always behave as if it is in the mode it was in during tracing, no matter which mode the
          `ScriptModule` is in..."

        In order to avoid modifying self during the save() call, we actually deepcopy self and then do the freeze and
        trace on the copy.
        """
        output_dir = os.path.split(filename)[0]
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        clone = copy.deepcopy(self)
        clone.to('cpu')
        clone.eval()
        example_input = torch.zeros((1, INPUT_SIZE))
        mod = torch.jit.trace(clone, example_input)
        mod.save(filename)
        if verbose:
            print(f'Model saved to {filename}')

    def add_head(self, head: nn.Module, target: LearningTarget):
        assert len(self.heads) == len(self.learning_targets)
        n = len(self.heads)
        if n == 0:
            assert target.name == 'output', 'The first target must be the output target'

        assert target.name not in self.target_names(
        ), f'Target with name {target.name} already exists'
        self.heads.append(head)
        self.learning_targets.append(target)

    def target_names(self) -> List[str]:
        return [target.name for target in self.learning_targets]

    @staticmethod
    def load_checkpoint(filename: str) -> 'NeuralNet':
        checkpoint = torch.load(filename)
        model_state_dict = checkpoint['model_state_dict']
        model = NeuralNet()
        model.load_state_dict(model_state_dict)
        return model

    @abc.abstractmethod
    def save_checkpoint(self, filename: str):
        torch.save({
            'model_state_dict': self.state_dict(),
        }, filename)
