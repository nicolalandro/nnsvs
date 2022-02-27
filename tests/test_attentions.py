import torch


def test_nonar_attn():
    from nnsvs.model import NonARSelfAttDecoder

    model = NonARSelfAttDecoder(
        in_dim=419,
        out_dim=199,
        hidden_channels=256,
        filter_channels=256,
        n_heads=4,
        n_layers=2,
    )

    inputs = torch.rand(2, 120, 419)
    in_lens = torch.tensor([120, 120], dtype=torch.long, device=inputs.device)
    out = model(inputs, in_lens)

    print(out.shape)
    assert out.shape == (2, 120, 199)
