from src.TextSummarizer.constants import *
import unittest

# Compatibility fix for libraries using the removed Python 2/older Python API
if not hasattr(unittest.TestCase, "assertRaisesRegexp"):
    unittest.TestCase.assertRaisesRegexp = unittest.TestCase.assertRaisesRegex

from src.TextSummarizer.utils.common import read_yaml, create_directories
from src.TextSummarizer.entity import DataIngestionConfig,DataTransformationConfig


# Every component has a funcionality which need to be defined

class ConfigurationManager:
    def __init__(self,
                 config_path =  CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH):
        self.config=read_yaml(config_path)
        self.params=read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) ->  DataIngestionConfig:
        config=self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config=self.config.data_transformation
        create_directories([config.root_dir])

        data_ingestion_config = DataTransformationConfig(
            root_dir = config.root_dir,
            data_path = config.data_path,
            tokenizer_name = config.tokenizer_name
        )

        return data_ingestion_config


