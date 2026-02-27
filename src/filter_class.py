from core.data_manager import DataProcessor

dp = DataProcessor("/home/team_thuctap/ltlinh/new_ndanh/data/merged_data/v3_filterBB/data.yaml")
dp.filter_class(
    class_name="unidentified",
    splits=["train", "val"],
    new_dataset_root="data/merged_data/v4_filterUnidentified"
)
