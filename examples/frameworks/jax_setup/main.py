import jax
import jax.extend.backend


def main():
    device_count = len(jax.local_devices(backend="gpu"))
    print(f"Jax default backend:         {jax.extend.backend.get_backend().platform}")
    print(f"Jax-detected #GPUs:          {device_count}")

    if device_count == 0:
        print("    No GPU detected, not printing device names.")
    else:
        for i, device in enumerate(jax.local_devices(backend="gpu")):
            print(f"    GPU {i}:      {device.device_kind}")


if __name__ == "__main__":
    main()
