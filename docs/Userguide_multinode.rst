Multiple Nodes
==============

Data Parallel
-------------

.. image:: _static/dataparallel.png

Request 3 nodes with each at least 4 GPUs each.

.. code-block:: bash
   :linenos:

    # Number of Nodes
    #SBATCH --nodes=3

    # Number of tasks. 3 (1 per node)
    #SBATCH --ntasks=3

    # Number of GPU per node
    #SBATCH --gres=gpu:4
    #SBATCH --gpus-per-node=4

    # 16 CPUs per node
    #SBATCH --cpus-per-gpu=4

    # 16Go per nodes (4Go per GPU)
    #SBATCH --mem=16Go

    # we need all nodes to be ready at the same time
    #SBATCH --wait-all-nodes=1

    # Total resources:
    #   CPU: 16 * 3 = 48
    #   RAM: 16 * 3 = 48 Go
    #   GPU:  4 * 3 = 12

    # Setup our rendez-vous point
    RDV_ADDR=$(hostname)
    WORLD_SIZE=$SLURM_JOB_NUM_NODES
    # -----

    srun -l torchrun \
        --nproc_per_node=$SLURM_GPUS_PER_NODE\
        --nnodes=$WORLD_SIZE\
        --rdzv_id=$SLURM_JOB_ID\
        --rdzv_backend=c10d\
        --rdzv_endpoint=$RDV_ADDR\
        training_script.py


.. code-block:: python

    import os
    import torch.distributed as dist


    class Trainer:
        def __init__(self):
            self.local_rank = None
            self.chk_path = None

        @property
        def device_id(self):
            return self.local_rank

        def load_checkpoint(self, path):
            self.chk_path = path
            # ...

        def initialize(self):
            self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
            if self.local_rank < 0:
                raise RuntimeError(f"Could not find LOCAL_RANK")

            dist.init_process_group(backend="gloo|nccl")

        def dataset(self):
            train_sampler = ElasticDistributedSampler(train_dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
                sampler=train_sampler,
            )
            return train_loader

        def save_checkpoint(self):
            if self.chk_path is None:
                return

        def train_step(self):
            pass

        def train(self):
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.device_id],
                output_device=self.device_id
            )

            dataset = self.dataset()

            for epoch in range(100):
                for batch in iter(dataset):
                    self.train_step(batch)

                    if should_checkpoint:
                        self.save_checkpoint()

    def main():
        trainer = Trainer()
        trainer.load_checkpoint(path)
        tainer.initialize()
        trainer.train()

.. note::

    To bypass Python GIL (Global interpreter lock) pytorch spawn one process for each GPU.
    In the example above this means at least 12 processes are spawn, at least 4 on each node.
