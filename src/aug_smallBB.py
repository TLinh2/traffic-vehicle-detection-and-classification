from core.data_manager import DataProcessor

if __name__ == "__main__":
    dp = DataProcessor("data/v1_2_filteredBB/data.yaml")

    new_dataset_root = "data/v1_3_augmentBB"
    dp.augment_small_objects(new_dataset_root=new_dataset_root)
    