##
##
##

import os
from time import strftime
import yaml
from yaml import SafeLoader
from munch import DefaultMunch
from typing import List

import src.utils as utils
from src.dataset.config import DatasetConfig, DatasetConfigBuilder
from src.model.config import ModelConfig
from src.training.optimizer import OptimizerConfig
from src.training.lr_scheduler import LRSchedulerConfig
from src.test.config import TestConfig
from src.training.config import TrainingConfig

class Config:
    
    def __init__(self, 
                 config_file: str,
                 generate: bool,
                 train: bool, 
                 test: bool, 
                 debug: bool) -> None:
        
        options = Config._get_options(config_file)
        
        self.debug = debug
        
        self.work_dir = options.work_dir
        if self.work_dir is None:
            self.work_dir = './work_dir'
            
        self.work_dir = os.path.join(self.work_dir, strftime('%Y-%m-%d-%H-%M-%S'))
        utils.check_and_create_dir(self.work_dir)
        
        config_name = os.path.splitext(config_file)[0]
        config_name = os.path.basename(config_name)
        self.log_file = os.path.join(self.work_dir, f'{config_name}.log')
        
        self.gpus = options.gpus
        self.seed = options.seed
        
        self.datasets_config = Config._get_datasets(options, generate)
        self.models_config = Config._get_models(options, train, test)
        self.optimizers_config = Config._get_optimizers(options, train)
        self.lr_schedulers_config = Config._get_lr_schedulers(options, train)
        self.trainings_config = self._get_trainings(options, train)
        self.tests_config = self._get_tests(options, test) 
    
    @staticmethod
    def _get_options(config_file):
        if not os.path.isfile(config_file):
            raise ValueError(f'Config file {config_file} does not exist.')

        try:
            with open(config_file, 'rb') as f: 
                options = yaml.load(f, SafeLoader)
        except:
            raise(f'An error occurred while trying to read config file.')
        
        options = DefaultMunch.fromDict(options, default=None)
        
        return options

    @staticmethod
    def _get_datasets(options: dict, generate: bool) -> List[DatasetConfig]:
        dataset_options = options.datasets
        if dataset_options is None: 
            raise ValueError('No config options for datasets were provided.')
        
        if type(dataset_options) != list: 
            raise ValueError('Configurations for datasets must be a list.')
        
        datasets = []
        for opt in dataset_options:
            datasets.append(DatasetConfigBuilder.build(opt, generate))
        
        return datasets
    
    @staticmethod
    def _get_models(options: dict, train: bool, test: bool) -> List[ModelConfig]:
        if not train and not test:
            return []
        
        model_options = options.models
        if model_options is None:
            raise ValueError('No config options for models were provided.')

        if type(model_options) != list: 
            raise ValueError('Configurations for models must be a list.')
        
        models = []
        for opt in model_options:
            models.append(ModelConfig(opt))
        
        return models

    @staticmethod
    def _get_optimizers(options: dict, train: bool) -> List[OptimizerConfig]:
        if not train:
            return []
        
        optimzer_options = options.optimizers
        if optimzer_options is None:
            raise ValueError('No config options for optimizers were provided.')

        if type(optimzer_options) != list: 
            raise ValueError('Configurations for optimizers must be a list.')
        
        optimizers = []
        for opt in optimzer_options:
            optimizers.append(OptimizerConfig(opt))
        
        return optimizers

    @staticmethod
    def _get_lr_schedulers(options: dict, train: bool) -> List[LRSchedulerConfig]:
        if not train:
            return []
        
        scheduler_options = options.lr_schedulers
        if scheduler_options is None:
            raise ValueError('No config options for lr schedulers were provided.')

        if type(scheduler_options) != list: 
            raise ValueError('Configurations for lr schdulers must be a list.')
        
        schedulers = []
        for opt in scheduler_options:
            schedulers.append(LRSchedulerConfig(opt))
        
        return schedulers
    
    def _get_trainings(self, options: dict, train: bool) -> List[TrainingConfig]:
        if not train: 
            return []

        training_options = options.trainings
        if training_options is None: 
            raise ValueError('No config options for trainings were provided.')

        if type(training_options) != list: 
            raise ValueError('Configurations for trainings must be a list.')

        trainings = []
        for idx, opt in enumerate(training_options): 
            if opt.work_dir is None:
                opt.work_dir = self.work_dir
            if opt.gpus is None:
                opt.gpus = self.gpus
            if opt.seed is None: 
                opt.seed = self.seed
            opt.debug = self.debug
            trainings.append(TrainingConfig(opt, 
                                            self.datasets_config, 
                                            self.models_config,
                                            self.optimizers_config, 
                                            self.lr_schedulers_config, 
                                            idx))
        
        return trainings

    def _get_tests(self, options: dict, test: bool) -> List[TestConfig]:
        if not test:
            return []

        test_options = options.tests
        if test_options is None: 
            raise ValueError('No config options for tests were provided.')

        if type(test_options) != list: 
            raise ValueError('Configurations for tests must be a list.')

        tests = []
        for opt in test_options: 
            if opt.work_dir is None:
                opt.work_dir = self.work_dir
            if opt.gpus is None:
                opt.gpus = self.gpus
            if opt.seed is None: 
                opt.seed = self.seed
            opt.debug = self.debug
            tests.append(TestConfig(opt,
                                    self.datasets_config,
                                    self.models_config))
        
        return tests
    