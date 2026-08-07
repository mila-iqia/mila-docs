import torch
from flash_attn import flash_attn_interface


def main():
    assert torch.cuda.is_available(), "Flash Attention requires a CUDA-capable GPU."
    device = torch.device("cuda")

    # Flash Attention requires float16 or bfloat16 inputs.
    dtype = torch.bfloat16

    batch_size = 2
    seq_len = 4096
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

    out, _ = flash_attn_interface.flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
    )

    print(f"GPU:          {torch.cuda.get_device_name(device)}")
    print(f"Input  shape: {q.shape}  dtype: {q.dtype}")
    print(f"Output shape: {out.shape}  dtype: {out.dtype}")


if __name__ == "__main__":
    main()
