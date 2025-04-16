"""the main file of the project"""

import logging
import os
import warnings

from data_augmentation import DataAugmentation
from recognizer import Recognizer
from ultrafarma_scrapper import UltrafarmaScraper

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

tf_logger = logging.getLogger("tensorflow")
tf_logger.setLevel(logging.FATAL)

if __name__ == "__main__":
    """
    scraper = UltrafarmaScraper()
    medicines_list = ['dipirona', 'ibuprofeno', 'paracetamol']
    total = 0
    for medicine in medicines_list:
        total += scraper.fetch_images(medicine)
    print(f"Total images downloaded: {total}")
    augmenter = DataAugmentation()
    summary = augmenter.augment_all_classes(num_augmented_images=40)
    print(summary)
    """
    recognizer = Recognizer()
    recognizer.run()
