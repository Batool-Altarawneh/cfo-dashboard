"""
Source Connector - ETL Extract Layer

This module is responsible for reading raw source files into pandas DataFrames.
"""

import os
from datetime import datetime

import pandas as pd


# ---------------------------------------------------------------------------
# Supported file types
# ---------------------------------------------------------------------------

# These are the only file extensions this connector is allowed to read.

SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".csv"}


def read_source_file(file_path: str) -> tuple[pd.DataFrame, dict]:
    """
    Read a raw source file and return both the data and file metadata.

    Parameters
    ----------
    file_path : str
        The path to the source file.
        Example: "data/raw/transactions.xlsx"

    Returns
    -------
    tuple[pd.DataFrame, dict]
        A tuple containing:
        1. df: The raw data as a pandas DataFrame.
        2. metadata: A dictionary with information about the source file.

    """

    # -----------------------------------------------------------------------
    # Step 1: Check that the file exists
    # -----------------------------------------------------------------------

    # Before pandas tries to read the file, we check if the path is valid.
    # This gives a clear error message if the file name or folder is wrong.
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found: {file_path}")

    # -----------------------------------------------------------------------
    # Step 2: Detect the file extension
    # -----------------------------------------------------------------------


    _, extension = os.path.splitext(file_path)

    # Convert the extension to lowercase so ".XLSX" and ".xlsx" are treated the same way.
    extension = extension.lower()

    # -----------------------------------------------------------------------
    # Step 3: Make sure the file type is supported
    # -----------------------------------------------------------------------

    # If the file is not Excel or CSV, the connector should stop immediately.
    # This prevents unexpected file types from entering the ETL pipeline.
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: '{extension}'. "
            f"Supported file types are: {SUPPORTED_EXTENSIONS}"
        )

    # -----------------------------------------------------------------------
    # Step 4: Read the file using the correct pandas function
    # -----------------------------------------------------------------------

    if extension in {".xlsx", ".xls"}:
        df = _read_excel(file_path)
    elif extension == ".csv":
        df = _read_csv(file_path)
    else:
   
        raise ValueError(f"No reader configured for file type: {extension}")

    # -----------------------------------------------------------------------
    # Step 5: Create metadata about the loaded file
    # -----------------------------------------------------------------------

    # This metadata can later be added to the staging table.
    # That way, every row in PostgreSQL can be traced back to its original file.
    metadata = {
        # Only the file name, not the full path.
        # Example: "transactions.xlsx"
        "source_file": os.path.basename(file_path),

        # Full absolute path.
        # Useful when debugging because it shows exactly where the file came from.
        "source_path": os.path.abspath(file_path),

        # Load timestamp in UTC.
        # UTC is better than local time for data pipelines because it is consistent across machines, servers, and time zones.
        "loaded_at": datetime.utcnow().isoformat(),

        # Number of rows loaded from the file.
        "row_count": len(df),

        # Number of columns loaded from the file.
        "col_count": len(df.columns),

        # Original column names from the source file.
        # This is useful for schema validation in the next step.
        "columns": list(df.columns),
    }


    print(
        f" Read '{metadata['source_file']}': "
        f"{metadata['row_count']:,} rows × {metadata['col_count']} columns"
    )

    return df, metadata


def _read_excel(file_path: str) -> pd.DataFrame:
    """
    Read an Excel file into a pandas DataFrame.

    This is a helper function used only inside this module.

    
    """

    try:
        df = pd.read_excel(
            file_path,
            dtype=object,
            engine="openpyxl"
        )

        return df

    except Exception as error:

        raise RuntimeError(
            f"Failed to read Excel file '{file_path}': {error}"
        )


def _read_csv(file_path: str) -> pd.DataFrame:
    """
    Read a CSV file into a pandas DataFrame.

    This is also a helper function used only inside this module.
    """

    try:
        df = pd.read_csv(
            file_path,
            dtype=object,
            encoding="utf-8",

           
            on_bad_lines="warn"
        )

        return df

    except Exception as error:
        raise RuntimeError(
            f"Failed to read CSV file '{file_path}': {error}"
        )


def get_file_info(file_path: str) -> dict:
    """
    Return basic file information without reading the full file.

    This is useful when we want to log file details before loading it,
    especially if the file is large.

    Returns
    -------
    dict
        A dictionary containing:
        - filename
        - file size in KB
        - last modified timestamp
    """

    # Check the file exists before trying to get file system information.
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

   
    file_stats = os.stat(file_path)

    return {
        # File name only.
        "filename": os.path.basename(file_path),

        # File size is returned in bytes.
        # Dividing by 1024 converts it to KB.
        "size_kb": round(file_stats.st_size / 1024, 2),

        # Convert the last modified timestamp into a readable datetime format.
        "last_modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
    }


# ---------------------------------------------------------------------------
# Script entry point - quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
  
    import sys

    # If the user passes a file path in the terminal, use it.
    # Otherwise, use transactions.xlsx as the default test file.
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "data/raw/transactions.xlsx"

    # Print basic file information before loading the full file.
    file_info = get_file_info(path)

    print(
        f"File: {file_info['filename']} | "
        f"Size: {file_info['size_kb']} KB | "
        f"Last modified: {file_info['last_modified']}"
    )

    # Read the file and collect both the data and metadata.
    df, metadata = read_source_file(path)

    # Print the original columns so we can quickly confirm the file structure.
    print(f"Columns: {metadata['columns']}")

    # Print the first 3 rows as a quick visual check.
    print(df.head(3))