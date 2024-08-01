# ImageNet setup

echo "Setting up ImageNet directories and creating symlinks..."
mkdir -p $SLURM_TMPDIR/imagenet
ln -s /network/datasets/imagenet/ILSVRC2012_img_train.tar -t $SLURM_TMPDIR/imagenet 
ln -s /network/datasets/imagenet/ILSVRC2012_img_val.tar -t $SLURM_TMPDIR/imagenet
ln -s /network/datasets/imagenet/ILSVRC2012_devkit_t12.tar.gz -t $SLURM_TMPDIR/imagenet
echo "Creating ImageNet validation dataset..."
python -c "from torchvision.datasets import ImageNet; ImageNet('$SLURM_TMPDIR/imagenet', split='val')"
echo "Creating ImageNet training dataset..."
mkdir -p $SLURM_TMPDIR/imagenet/train
tar -xf /network/datasets/imagenet/ILSVRC2012_img_train.tar \
     --to-command='mkdir -p $SLURM_TMPDIR/imagenet/train/${TAR_REALNAME%.tar}; \
                    tar -xC $SLURM_TMPDIR/imagenet/train/${TAR_REALNAME%.tar}' \
     -C $SLURM_TMPDIR/imagenet/train

# SLOWER: Obtain ImageNet files using torch directly
#python -c "from torchvision.datasets import ImageNet; ImageNet('$SLURM_TMPDIR/imagenet', split='train')"