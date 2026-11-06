import os
import torchvision.datasets as datasets

def download():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    print(f"Downloading MNIST to {data_dir}...")
    datasets.MNIST(root=data_dir, train=True, download=True)
    datasets.MNIST(root=data_dir, train=False, download=True)
    print("Download completed successfully!")

if __name__ == '__main__':
    download()
