import torch


def test_internal_taco2_decoder():
    from nnsvs.tacotron.decoder import NoAttDecoder

    # (B, T, C)
    encoder_outs = torch.rand(2, 120, 512)
    decoder_targets = torch.rand(2, 120, 80)

    decoder = NoAttDecoder()

    out = decoder(encoder_outs, None, decoder_targets)

    print(out.shape)

    # NOTE: internal decoder returns tensor of shape (B, C, T) (not (B, T, C))
    assert out.shape == (2, 80, 120)


def test_noatt_taco2():
    from nnsvs.model import NoAttTacotron2

    model = NoAttTacotron2(
        in_dim=419, out_dim=199, encoder_hidden_dim=64, decoder_hidden_dim=32
    )
    inputs = torch.rand(2, 120, 419)
    decoder_targets = torch.rand(2, 120, 199)
    in_lens = torch.tensor([120, 120], dtype=torch.long, device=inputs.device)

    out = model(inputs, in_lens, decoder_targets)

    print(out.shape)
    assert out.shape == decoder_targets.shape

    out_inf = model.inference(inputs, in_lens)

    print(out_inf.shape)
    assert out_inf.shape == decoder_targets.shape
