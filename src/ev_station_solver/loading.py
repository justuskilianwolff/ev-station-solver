import logging
import os
from pathlib import Path
from typing import Literal

import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)


def load_locations(
    mode: Literal["large", "medium", "small"] = "medium",
) -> DataFrame:
    if mode == "large":
        file_name = "large_vehicle_locations.csv"
    elif mode == "medium":
        file_name = "medium_vehicle_locations.csv"
    elif mode == "small":
        file_name = "small_vehicle_locations.csv"
    else:
        raise ValueError(f"Mode {mode} not supported.")

    # path from root of this project
    path_root = os.path.join("locations", file_name)

    script_dir = Path(__file__).resolve().parents[2]

    # Join the script directory with the relative path of the data file
    data_file_path = os.path.join(script_dir, path_root)

    # Load the data
    df = pd.read_csv(data_file_path, delimiter=",")

    return df
