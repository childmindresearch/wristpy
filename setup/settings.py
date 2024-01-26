"""Constants for the setup module."""
import pathlib as pl

DIR_REPO = pl.Path(__file__).parent.parent
LICENSES = ("mit", "lgpl-2.1", "lgpl-3.0", "mpl-2.0", "unlicense")
TARGET_EXTENSIONS = {
    ".py",
    ".md",
    ".yml",
    ".yaml",
    ".toml",
    ".txt",
}
