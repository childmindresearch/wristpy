"""Functions for fetching and replacing the license file."""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Optional
from urllib import request

from setup import settings

DIR_REPO = settings.DIR_REPO
LICENSES = settings.LICENSES


def get_license(name: str) -> dict[str, str]:
    """Fetches the license files from GitHub.

    Args:
        name: The name of the license to fetch.

    Returns:
         The license text.

    """
    if name not in LICENSES:
        raise ValueError(f"License {name} is not in {LICENSES}.")
    with request.urlopen(f"https://api.github.com/licenses/{name}") as response:
        license_info = response.read().decode()
    return json.loads(license_info)


def modify_license_placeholder_text(selected_license: dict[str, str]) -> dict[str, str]:
    """Modifies the placeholder text in the license.

    Args:
        selected_license: The license to modify.

    Returns:
        The modified license.

    """
    if selected_license["key"] == "mit":
        license_holder = input("Who is the holder of the license? ")
        current_year = datetime.now().year
        selected_license["body"] = selected_license["body"].replace(
            "[year]",
            str(current_year),
        )
        selected_license["body"] = selected_license["body"].replace(
            "[fullname]",
            license_holder,
        )

    return selected_license


def request_license() -> Optional[dict[str, str]]:
    """Asks the user to select a license.

    Returns:
        The path to the selected license.

    """
    print("Available licenses:")
    print("\t0. No license")
    for i, option in enumerate(LICENSES):
        print(f"\t{i + 1}. {option}")
    while True:
        try:
            choice = int(input("Enter the number of the license you want to use: "))
            if choice == 0:
                return None
            if 0 < choice <= len(LICENSES):
                selected_license = get_license(LICENSES[choice - 1])
                break
            raise ValueError("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid choice. Please try again.")

    final_license = modify_license_placeholder_text(selected_license)
    return final_license


def replace_license_badge(content: str, repo_license: Optional[dict[str, str]]) -> str:
    """Replaces the license badge with the specified license.

    Args:
        content: Content of any file that might contain a license badge.
        repo_license: The license to replace the current license with. If None,
        the current license badge will be deleted.

    Returns:
        The content with the license badge replaced.
    """
    if repo_license is None:
        # remove line containing license badge
        return re.sub(
            r"\[!\[.*License\]\(https://img\.shields\.io/badge/license.*\)\]\(.*\)\n",
            "",
            content,
        )
    # shield.io uses -- as an escape character, so we need to replace - with --
    license_name_upper = repo_license["key"].upper().replace("-", "--")
    return content.replace(
        "![MIT License]" "(https://img.shields.io/badge/license-MIT-blue.svg)]",
        f"![{license_name_upper} License]"
        f"(https://img.shields.io/badge/license-{license_name_upper}-blue.svg)]",
    )


def replace_license(repo_license: Optional[dict[str, str]]) -> None:
    """Replaces the license file in the repository with the specified license.

    Args:
        repo_license: The license to replace the current license with. If None,
        the current license file will be deleted.

    """
    license_file = DIR_REPO / "LICENSE"
    license_file.unlink(missing_ok=True)

    if repo_license is None:
        return

    with open(license_file, "w", encoding="utf-8") as file_buffer:
        file_buffer.write(repo_license["body"])
