##
##
##

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .. import utils

class LRSchedulerConfig:
    def __init__(self, options: dict) -> None:
        self._name = options.name
        
        if self._name is None:
            raise ValueError(f'Missing name for lr scheduler.')
        
        if not utils.check_class_exists('torch.optim.lr_scheduler', self._name):
            raise ValueError(f'No lr scheduler with the given name was found.')
        
        del options.name 
        self._args = options
        
    def to_lr_scheduler(self, optimizer: Optimizer) -> _LRScheduler:
        cls =  utils.get_class_by_name('torch.optim.lr_scheduler', self._name)
        return cls(optimizer=optimizer, **self._args)