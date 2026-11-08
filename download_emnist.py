import os
import torchvision.datasets as datasets

def download_emnist():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    print(f"Downloading EMNIST ByClass dataset to {data_dir}...")
    # This downloads gzip.zip (~535MB) and extracts it into data/EMNIST/raw
    datasets.EMNIST(root=data_dir, split='byclass', train=True, download=True)
    datasets.EMNIST(root=data_dir, split='byclass', train=False, download=True)
    print("EMNIST ByClass downloaded and prepared successfully!")

if __name__ == '__main__':
    download_emnist()

