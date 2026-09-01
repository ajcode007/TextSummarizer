import os
from src.TextSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_from_disk
from src.TextSummarizer.config.configuration import DataTransformationConfig


class DataTransformation:
    def __init__(self,config :DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self,example_batch):
        input_encodings = self.tokenizer(example_batch['dialogue'] , max_length = 1024, truncation = True )

        target_encodings = self.tokenizer(example_batch['summary'], max_length = 128, truncation = True )

        labels = target_encodings['input_ids']

        # Replace padding token id in the labels with -100 to be ignored by the loss function
        labels = [[(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels]


        return {
        'input_ids' : input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': labels
    }

    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched = True)
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir,"samsum_dataset"))
