import os
import tarfile
import subprocess

# Paths
train_dir = "/argotrain"
test_dir = "/argotest"
train_tar = os.path.join(train_dir, "train-000.tar")
test_tar = os.path.join(test_dir, "test-000.tar")


os.makedirs(train_dir, exist_ok=True)
# os.makedirs(test_dir, exist_ok=True)  # Uncomment if you want test dir too

# Download Train 1 with aria2c
train_url = "https://s3.amazonaws.com/argoverse/datasets/av2/tars/sensor/train-000.tar"
if not os.path.exists(train_tar):
    print("Downloading training data...")
    subprocess.run([
        "aria2c", "-x", "16", train_url, "-d", train_dir, "-o", "train-000.tar"
    ], check=True)
else:
    print("Train tar already exists, skipping download.")

# Download Test 1 if needed
# test_url = "https://s3.amazonaws.com/argoverse/datasets/av2/tars/sensor/test-000.tar"
# if not os.path.exists(test_tar):
#     print("Downloading test data...")
#     subprocess.run([
#         "aria2c", "-x", "16", test_url, "-d", test_dir, "-o", "test-000.tar"
#     ], check=True)

# Extract tar file
print("Extracting training data...")
with tarfile.open(train_tar, "r") as tar:
    tar.extractall(train_dir)

print("Done!")
## Will take up to 5-10 mins

