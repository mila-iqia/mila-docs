import os
import torch
from   torchvision.datasets   import CIFAR10
from   torchvision.transforms import ToTensor


def main():
    # Check GPU is available
    assert(torch.cuda.is_available() and
           torch.cuda.device_count() > 0)
    
    # Obtain CIFAR10
    device = torch.device("cuda", 0)
    dataset_path = os.environ.get('SLURM_TMPDIR', '../dataset')
    dataset = CIFAR10(dataset_path, transform=ToTensor())
    dataset_len = len(dataset)
    
    # Perform sum and average on a single GPU
    dataset_sum = torch.zeros_like(dataset[0][0], dtype=torch.float32, device=device)
    for i in range(dataset_len):
        dataset_sum.add_(dataset[i][0].to(dtype=torch.float32, device=device))
    dataset_avg = dataset_sum.mul_(255.0/dataset_len).to(dtype=torch.uint8, device="cpu")
    
    # Print
    print("Average of CIFAR10 Training Dataset:")
    for cnum, c in enumerate(dataset_avg):
        print(f'Channel {cnum}')
        for r in c:
            print(''.join([f'{int(v):>4d}' for v in r]))


if __name__ == "__main__":
    main()
