"""Make sure the data is available"""
import sys
import time

from torchvision.datasets import INaturalist


start_time = time.time()
INaturalist(root=sys.argv[1], version="2021_train", download=True)
INaturalist(root=sys.argv[1], version="2021_valid", download=True)
seconds_spent = time.time() - start_time
print(f"Prepared data in {seconds_spent/60:.2f}m")
