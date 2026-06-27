"""
Utilities specific to the APL / Aragon Photonics HDAS 2.5 interrogator.
"""

import os
import re
from datetime import datetime


def get_datetime_from_filename(file_path: str) -> datetime:
    """Extract the UTC timestamp from an HDAS 2.5 filename.

    Aragon Photonics encodes the acquisition start time in the filename
    using the pattern:  YYYY_MM_DD_HHhMMmSSs_...

    Args:
        file_path: Filename or full path to the .bin file.

    Returns:
        datetime object representing the timestamp in the filename.

    Raises:
        ValueError: If the filename does not match the expected pattern.
    """
    filename = os.path.basename(file_path)
    pattern = r'(\d{4})_(\d{2})_(\d{2})_(\d{2})h(\d{2})m(\d{2})s'
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(f"No valid timestamp found in filename: {filename!r}")
    year, month, day, hour, minute, second = map(int, match.groups())
    return datetime(year, month, day, hour, minute, second)
