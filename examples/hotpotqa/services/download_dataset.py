import os
import subprocess
import hashlib
from pathlib import Path

# Define constants
DATASET_DIR = Path(__file__).parent.parent / "dataset/HotpotQA"
FILE_URL = "https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
FILE_NAME = "enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
EXPECTED_MD5 = "01edf64cd120ecc03a2745352779514c"

# Create the dataset directory
os.makedirs(DATASET_DIR, exist_ok=True)

# Download the file
file_path = os.path.join(DATASET_DIR, FILE_NAME)
print("Downloading the dataset...")
subprocess.run(["curl", "-O", FILE_URL, "--output-dir", DATASET_DIR], check=True)

# Verify the MD5 checksum
def calculate_md5(file_path):
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()

print("Verifying the MD5 checksum...")
actual_md5 = calculate_md5(file_path)
if actual_md5 == EXPECTED_MD5:
    print("MD5 checksum matches. Extracting the dataset...")
else:
    print("MD5 checksum does not match. Please try downloading again.")
    exit(1)

# Extract the tar.bz2 file
subprocess.run(["tar", "-xjvf", file_path, "-C", DATASET_DIR], check=True)

# Clean up
os.remove(file_path)
print("Dataset downloaded, verified, extracted, and cleaned up successfully.")

# Download the HotpotQA Test and Dev sets
DEV_SET_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"
TEST_SET_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json"

def download_file(url, output_dir):
    subprocess.run(["curl", "-O", url, "--output-dir", output_dir], check=True)

print("Downloading the HotpotQA Test and Dev sets...")
download_file(DEV_SET_URL, DATASET_DIR)
download_file(TEST_SET_URL, DATASET_DIR)
print("HotpotQA Test and Dev sets downloaded successfully.")
