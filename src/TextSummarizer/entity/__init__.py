from dataclasses import dataclass
from pathlib import Path

# define the what kind of input and ouput for that particular component 

@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_URL: Path
    local_data_file: Path
    unzip_dir: Path

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: Path

@dataclass
class ModelTrainingConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: Path
    num_train_epochs : int
    warmup_steps : int
    per_device_train_batch_size : int
    weight_decay : float
    logging_steps : int
    eval_strategy: str
    eval_steps : int
    save_steps : float
    gradient_accumulation_steps : int