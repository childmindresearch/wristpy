#!/usr/bin python3
"""Setup template for Python repositories."""
import os
import pathlib as pl
import shutil

from setup import licenses, settings

DIR_REPO = settings.DIR_REPO
TARGET_EXTENSIONS = settings.TARGET_EXTENSIONS


def main() -> None:
    """Entrypoint to the template setup script.

    This script will ask the user for details of the repository and then replace
    the template values with the user input. It will also remove the setup files
    and the setup directory.
    """
    # Collect some data
    git_uncommitted_changes = (
        os.popen(f"git -C {DIR_REPO} status -s").read().strip() != ""
    )
    git_username = os.popen(f"git -C {DIR_REPO} config user.name").read().strip()
    git_email = os.popen(f"git -C {DIR_REPO} config user.email").read().strip()
    git_repo_name = (
        os.popen(f"git -C {DIR_REPO} remote get-url origin")
        .read()
        .split("/")[-1]
        .split(".")[0]
    )

    # Ask for some data
    if git_uncommitted_changes:
        print("You have uncommitted changes. Please commit or stash them first.")
        exit(1)
    repo_name = (
        input(f"Enter the name of the repository [{git_repo_name}]: ") or git_repo_name
    )
    module_name = input(f"Enter the name of the module [{repo_name}]: ") or repo_name
    username = input(f"Enter your username [{git_username}]: ") or git_username
    email = input(f"Enter your email [{git_email}]: ") or git_email
    description = (
        input("Enter a short description of the project: ")
        or "A beautiful description."
    )
    repo_license = licenses.request_license()

    # Print the data
    print(
        f"Using the following values:\n"
        f"\tRepository name: '{repo_name}'\n"
        f"\tModule name: '{module_name}'\n"
        f"\tAuthor: '{username} <{email}'>\n"
        f"\tDescription: '{description}'\n"
        f"\tLicense: '{repo_license['name'] if repo_license else 'No license'}'",
    )
    input("Press enter to continue...")

    # Replace the template values
    for file in pl.Path(DIR_REPO).glob("**/*"):
        if (
            not file.is_file()
            or file.suffix not in TARGET_EXTENSIONS
            or file.name == "setup_template.py"
        ):
            continue

        with open(file, encoding="utf-8") as f:
            content = f.read()

        content_before = content
        content = content.replace(
            "- [ ] Run `setup_template.py`",
            "- [x] Run `setup_template.py`",
        )
        content = content.replace(
            "- [ ] Update the `LICENSE`",
            "- [x] Update the `LICENSE`",
        )
        content = content.replace("template-python-repository", repo_name)
        content = content.replace("APP_NAME", module_name)
        content = content.replace("app-name", module_name)
        content = content.replace("A beautiful description.", description)
        content = content.replace("reinder.vosdewael@childmind.org", email)
        content = content.replace("ENTER_YOUR_EMAIL_ADDRESS", email)
        content = content.replace("Reinder Vos de Wael", username)

        content = licenses.replace_license_badge(content, repo_license)

        if content != content_before:
            print(f"Updating {file.relative_to(DIR_REPO)}")
            with open(file, "w", encoding="utf-8") as f:
                f.write(content)

    licenses.replace_license(repo_license)

    dir_module = DIR_REPO / "src" / "APP_NAME"
    if dir_module.exists():
        dir_module.rename(dir_module.parent / module_name)

    # Remove setup files
    print("Removing setup files.")
    setup_files = pl.Path(DIR_REPO / "setup").glob("*.py")
    for setup_file in setup_files:
        pl.Path(DIR_REPO / "setup" / setup_file).unlink()
    if pl.Path(DIR_REPO / "setup" / "__pycache__").exists():
        # Use a more robust method to remove the cache directory
        shutil.rmtree(DIR_REPO / "setup" / "__pycache__")
    pl.Path(DIR_REPO / "setup").rmdir()
    pl.Path(__file__).unlink()


if __name__ == "__main__":
    main()
