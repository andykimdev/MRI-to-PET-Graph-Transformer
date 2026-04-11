"""
Pathnames for input and output for preprocessing
"""
from pathlib import Path

# Pathname for project root (automatically detects file location)
ROOT = Path(__file__).resolve().parents[1]

print(ROOT)
