import torch

from models.fusion.cross_attention import CrossAttentionAdapter


def test_cross_attention_shapes():

    B = 2
    T_text = 32
    T_audio = 50
    d = 768

    text_hidden = torch.randn(B, T_text, d)
    audio_hidden = torch.randn(B, T_audio, d)

    module = CrossAttentionAdapter(
        hidden_dim=d,
        num_heads=8,
    )

    fused_hidden, attn_weights = module(
        text_hidden,
        audio_hidden,
    )

    assert fused_hidden.shape == (B, T_text, d)
    assert attn_weights.shape == (B, T_text, T_audio)
