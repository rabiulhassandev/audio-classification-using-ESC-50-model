import os
import urllib.request
import zipfile
import shutil

def download_esc50(target_dir="data"):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    zip_path = os.path.join(target_dir, "ESC-50-master.zip")
    
    print(f"Downloading ESC-50 dataset from {url}...")
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete.")
    
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    print("Extraction complete.")
    
    # Rename folder to remove branch name if strictly needed, but config uses ESC-50-master
    print(f"Dataset ready at {os.path.join(target_dir, 'ESC-50-master')}")

if __name__ == "__main__":
    download_esc50()
