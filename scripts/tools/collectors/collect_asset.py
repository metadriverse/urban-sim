import argparse
import logging
import os
import shutil
import zipfile
import time
import urllib.request
from pathlib import Path
import subprocess

import filelock
import progressbar
from filelock import Timeout

ROOT_DIR = Path(__file__).parent.parent.parent.parent
print(f"ROOT_DIR: {ROOT_DIR}")
file_id = "1YRA1R0GJMqOe0QRESPxkrvJzetvNL31-"
# ASSET_URL = f"https://drive.google.com/uc?export=download&id={file_id}"
# ASSET_URL = "https://drive.google.com/file/d/1YRA1R0GJMqOe0QRESPxkrvJzetvNL31-/view?usp=sharing"
ASSET_URL = "https://huggingface.co/datasets/Hollis71025/URBAN-SIM-Assets/resolve/main/assets_urbansim.zip?download=true"
print(f"ASSET_URL: {ASSET_URL}")                                            


class MyProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()

def wait_asset_lock():
    print(
        f"[INFO] Another instance of this program is already running. "
        "Wait for the asset pulling finished from another program..."
    )

def pull_asset():

    assets_folder = ROOT_DIR / "assets"
    zip_path = ROOT_DIR / 'assets.zip'
    lock_path = ROOT_DIR / 'assets.lock'
    temp_assets_folder = ROOT_DIR / "temp_assets"

    lock = filelock.FileLock(lock_path, timeout=1)

    # Download the file
    try:
        with lock:
            import gdown
            # Download assets
            print(f"[INFO] Thank you for using URBAN-SIM! We would download assets for you.")
            print("[INFO] Pull assets from {} to {}".format(ASSET_URL, zip_path))
            # gdown.download(ASSET_URL, str(zip_path), fuzzy=True, quiet=False, use_cookies=True)
            os.system(f"wget --no-check-certificate {ASSET_URL} -O {zip_path}")

            # Prepare for extraction
            if os.path.exists(assets_folder):
                print("[INFO] Remove existing assets. Files: {}".format(os.listdir(assets_folder)))
                shutil.rmtree(assets_folder, ignore_errors=True)
            if os.path.exists(temp_assets_folder):
                shutil.rmtree(temp_assets_folder, ignore_errors=True)

            # Extract to temporary directory
            print(f"[INFO] Extracting assets.")
            shutil.unpack_archive(filename=str(zip_path), extract_dir=temp_assets_folder)
            shutil.move(str(temp_assets_folder / 'assets'), str(ROOT_DIR))

    except Timeout:  # Timeout will be raised if the lock can not be acquired in 1s.
        print(f"[INFO] Another instance of this program is already running. "
            "Wait for the asset pulling finished from another program..."
        )
        wait_asset_lock()
        print(f"[INFO] Assets are now available.")

    finally:
        # Cleanup
        for path in [zip_path, lock_path, temp_assets_folder]:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)

    # Final check
    if not assets_folder.exists():
        raise ValueError("Assets folder does not exist! Files: {}".format(os.listdir(ROOT_DIR)))

    print(f"[INFO] Successfully download assets")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    pull_asset()
    