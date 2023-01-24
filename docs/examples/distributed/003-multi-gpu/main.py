import os
import torch
import torch.distributed
from   torchvision.datasets   import CIFAR10
from   torchvision.transforms import ToTensor


def main():
    # Check distributed available
    assert(torch.distributed.is_available())
    print('PyTorch Distributed available.')
    print('  Backends:')
    print(f'    Gloo: {torch.distributed.is_gloo_available()}')
    print(f'    NCCL: {torch.distributed.is_nccl_available()}')
    print(f'    MPI:  {torch.distributed.is_mpi_available()}')
    
    # Check GPU is available
    assert(torch.cuda.is_available() and
           torch.cuda.device_count() > 0)
    
    # Initialize distributed
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS_PER_NODE", "1"))),
        rank      =int(os.environ.get("RANK",       os.environ.get("SLURM_PROCID",          "0"))),
    )
    
    # Obtain CIFAR10.
    #   Only master (rank-0) downloads the dataset if that's needed!
    is_master = torch.distributed.get_rank() == 0
    device = torch.device("cuda", 0)
    dataset_path = os.environ.get('SLURM_TMPDIR', '../dataset')
    if is_master:  # Download THEN Barrier
        dataset = CIFAR10(dataset_path, transform=ToTensor(), download=is_master)
        torch.distributed.barrier()
    else:          # Barrier  THEN *NO* Download!
        torch.distributed.barrier()
        dataset = CIFAR10(dataset_path, transform=ToTensor(), download=is_master)
    dataset_len = len(dataset)
    
    # Perform partial sum on multiple GPU
    dataset_sum = torch.zeros_like(dataset[0][0], dtype=torch.float32, device=device)
    for i in range(rank, dataset_len, work_size):
        dataset_sum.add_(dataset[i][0].to(dtype=torch.float32, device=device))
    
    # Have partial sums in every process.
    # Execute collective sum-operation across nodes. Synchronizes!
    torch.distributed.all_reduce(dataset_sum, op=torch.distributed.ReduceOp.SUM)
    
    # Divide by dataset length
    dataset_avg = dataset_sum.mul_(255.0/dataset_len).to(dtype=torch.uint8, device="cpu")
    
    # Print, only on master.
    if is_master:
        print("Average of CIFAR10 Training Dataset:")
        for cnum, c in enumerate(dataset_avg):
            print(f'Channel {cnum}')
            for r in c:
                print(''.join([f'{int(v):>4d}' for v in r]))


if __name__ == "__main__":
    main()
