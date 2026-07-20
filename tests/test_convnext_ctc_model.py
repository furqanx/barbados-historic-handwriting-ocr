import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from src.ctc.models.convnext import ConvNeXtCTCConfig, ConvNeXtCTCModel


def test_convnext_ctc_forward_shape_without_pretraining() -> None:
    model = ConvNeXtCTCModel(
        ConvNeXtCTCConfig(
            vocab_size=12,
            pretrained=False,
            rnn_hidden_size=16,
            rnn_layers=1,
            dropout=0.0,
        )
    )
    images = torch.rand(2, 3, 64, 128)

    logits = model(images)

    assert logits.shape == (32, 2, 12)


def test_convnext_ctc_output_lengths_follow_backbone_reduction() -> None:
    model = ConvNeXtCTCModel(
        ConvNeXtCTCConfig(vocab_size=12, pretrained=False)
    )

    lengths = model.output_lengths(torch.tensor([128, 130, 3]))

    assert lengths.tolist() == [32, 32, 1]
