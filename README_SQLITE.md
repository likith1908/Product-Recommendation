SQLite import helper
====================

This workspace includes a small script to import all CSV files from `csv_files/` into an SQLite database named `data.db` in the repository root.

Files
- `scripts/to_sqllite.py` - script to import CSV files into `data.db`.

Usage
-----

1. (Optional) Create and activate a virtual environment.
2. Install pandas for better dtype inference and faster imports:

   pip install pandas

3. Run the script from repository root:

   python scripts/to_sqllite.py

Notes
- If `pandas` is not installed the script falls back to the standard library CSV reader and imports columns as TEXT.
- Table names are derived from CSV filenames and sanitized to lower_snake_case.
- Running the script will replace any existing tables with the same names.
