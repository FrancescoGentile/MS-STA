##
##
##

from typing import Optional

from .model import Model
from ..dataset.skeleton import SkeletonGraph

class ModelConfig:
    
    def __init__(self, options: dict) -> None:
        self._name = options.name
        self._pretrained_path = options.pretrained
    
    @property 
    def name(self) -> str: 
        return self._name
    
    @property
    def pretrained_path(self) -> Optional[str]:
        return self._pretrained_path
    
    def to_model(self, 
                 skeleton: SkeletonGraph, 
                 num_classes: int, 
                 num_frames: int) -> Model:
        return Model(self, skeleton, num_classes, num_frames)
