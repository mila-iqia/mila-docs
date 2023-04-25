"""Make sure the data is available"""
import sys
import time

from torchvision.datasets import INaturalist


t = -time.time()
INaturalist(root=sys.argv[1], version="2021_train", download=True)
INaturalist(root=sys.argv[1], version="2021_valid", download=True)
t += time.time()
print(f"Prepared data in {t/60:.2f}m")
