import os
import urllib.request as request
import zipfile
from src.TextSummarizer.logging import logger
from src.TextSummarizer.entity import DataIngestionConfig


class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename,header = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(f"File is downloaded")
        else:
            logger.info(f"File already exits")


    def extract_zip_file(self):
        """
        Zip_file_path:str
        Extract the zip file into the data directory
        Funtion return None
        """

        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path,exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file,'r') as zip_ref:
            zip_ref.extractall(unzip_path)