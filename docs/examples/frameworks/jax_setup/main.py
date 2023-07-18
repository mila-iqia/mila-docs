import jax
from jax.lib import xla_bridge


def main():
    device_count = len(jax.local_devices(backend="gpu"))
    print(f"Jax default backend:         {xla_bridge.get_backend().platform}")
    print(f"Jax-detected #GPUs:          {device_count}")

    if device_count == 0:
        print("    No GPU detected, not printing devices' names.")
    else:
        for i in range(device_count):
            print(f"    GPU {i}:      {jax.local_devices(backend='gpu')[0].device_kind}")


if __name__ == "__main__":
    main()
