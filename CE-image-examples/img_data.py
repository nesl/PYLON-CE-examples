'''
Codes are adapted from https://github.com/ML-KULeuven/deepproblog/blob/master/src/deepproblog/examples/MNIST/data/__init__.py
'''

import itertools
import json
import random
from pathlib import Path
from typing import Callable, List, Iterable, Tuple

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


_DATA_ROOT = Path(__file__).parent

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

datasets = {
    "train": torchvision.datasets.MNIST(
        root=str(_DATA_ROOT), train=True, download=True, transform=transform
    ),
    "test": torchvision.datasets.MNIST(
        root=str(_DATA_ROOT), train=False, download=True, transform=transform
    ),
}


class MNIST_Images(object):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, item):
        return datasets[self.subset][int(item[0])][0]


MNIST_train = MNIST_Images("train")
MNIST_test = MNIST_Images("test")


def addition(n: int, dataset: str, seed=None):
    """Returns a dataset for one-digit addition"""
    return MNISTOperator(
        dataset_name=dataset,
        function_name="addition",
        operator=sum,
        arity=n,
        seed=seed,
    )


def complex_pattern(n: int, dataset: str, prim_datasets: dict, fsm_list: list, seed=None):
    """Returns a dataset for complex event"""
    n_event_class = len(fsm_list) + 1 # Add a default event that does not satisfy any FSMs

    def complex_func(x: List[int]) -> int:
        """Generate event labels for a 3-atom sequence""" 
        for e in fsm_list:
            if e.check(x) is True: return e.label
        return n_event_class - 1
    
    return MNISTOperator(
        dataset_name=dataset,
        function_name="complex pattern",
        operator=complex_func,
        arity=n,
        seed=seed,
        datasets=prim_datasets,
    )


class MNISTOperator(Dataset):
    def __getitem__(self, index: int) -> Tuple[list, list, int]:
        l = self.data[index]
        label = self._get_label(index)
        l = [self.dataset[i][0] for i in l]
        return *l, label

    def __init__(
        self,
        dataset_name: str,
        function_name: str,
        operator: Callable[[List[int]], int],
        arity=2,
        seed=None,
        datasets=datasets,
    ):
        """Generic dataset for operator(img, img) style datasets.

        :param dataset_name: Dataset category to use (train, val, test)
        :param function_name: Name of PYLON constraint function
        :param operator: Operator to generate correct examples
        :param datasets: Dataset to use
        :param arity: Number of arguments for the operator
        :param seed: Seed for RNG
        """
        super(MNISTOperator, self).__init__()
        assert arity >= 1
        self.dataset_name = dataset_name
        self.dataset = datasets[self.dataset_name]
        self.function_name = function_name
        self.operator = operator
        self.arity = arity
        self.seed = seed
        mnist_indices = list(range(len(self.dataset)))
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(mnist_indices)
        dataset_iter = iter(mnist_indices)
        # Build list of examples (mnist indices)
        self.data = []
        try:
            while dataset_iter:
                self.data.append(
                    [
                        next(dataset_iter) for _ in range(self.arity)
                    ]
                )
        except StopIteration:
            pass

    def save_to_txt(self, filename=None):
        """
        Save to a TXT file (for one digit usage).

        Format is (EXAMPLE) for each line
        EXAMPLE :- ARGS,expected_result
        ARGS :- ONE_DIGIT_NUMBER,...
        ONE_DIGIT_NUMBER :- img_id
        """
        if filename is None:
            filename = self.dataset_name
        file = filename + ".txt"
        data_text = ['(' + ','.join(str(j) for j in self.data[i]) + ',' + str(self._get_label(i)) + ')' + '\n' for i in range(len(self))]
        with open(file, 'w') as txtfile:
            txtfile.writelines(data_text)

    def to_json(self):
        """
        Convert to JSON, for easy comparisons with other systems.

        Format is [EXAMPLE, ...]
        EXAMPLE :- [ARGS, expected_result]
        ARGS :- [MULTI_DIGIT_NUMBER, ...]
        MULTI_DIGIT_NUMBER :- [img_id, ...]
        """
        data = [(self.data[i], self._get_label(i)) for i in range(len(self))]
        return json.dumps(data)

    def _get_label(self, i: int):
        mnist_indices = self.data[i]
        # Figure out what the ground truth is, first map each parameter to the value:
        ground_truth = [
            self.dataset[i][1] for i in mnist_indices
        ]
        # Then compute the expected value:
        expected_result = self.operator(ground_truth)
        return expected_result

    def __len__(self):
        return len(self.data)
    

if __name__ == "__main__":
    data = addition(2,"train",seed=0)
    data.save_to_txt()
    print(data[0])