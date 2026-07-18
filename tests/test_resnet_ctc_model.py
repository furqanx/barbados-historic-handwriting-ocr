import pytest

torch = pytest.importorskip("torch")

from src.models.resnet_ctc import ResNetCTCConfig, ResNetCTCModel


def test_resnet_ctc_forward_shape() -> None:
    model = ResNetCTCModel(
        ResNetCTCConfig(
            vocab_size=12,
            base_channels=8,
            rnn_hidden_size=16,
            rnn_layers=1,
            dropout=0.0,
        )
    )
    images = torch.rand(2, 3, 64, 128)

    logits = model(images)

    assert logits.shape == (32, 2, 12)


def test_resnet_ctc_output_lengths_use_horizontal_stride_four() -> None:
    model = ResNetCTCModel(ResNetCTCConfig(vocab_size=12, base_channels=8))

    lengths = model.output_lengths(torch.tensor([128, 130, 3]))

    assert lengths.tolist() == [32, 32, 1]
