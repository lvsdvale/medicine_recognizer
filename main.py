"""the main file of the project"""

from data_augmentation import DataAugmentation
from ultrafarma_scrapper import UltrafarmaScraper

if __name__ == "__main__":
    """
    scraper = UltrafarmaScraper()
    medicines_list = ['dipirona', 'ibuprofeno', 'paracetamol']
    total = 0
    for medicine in medicines_list:
        total += scraper.fetch_images(medicine)
    print(f"Total images downloaded: {total}")
    """

    augmenter = DataAugmentation()
    summary = augmenter.augment_all_classes(num_augmented_images=40)
    print(summary)
