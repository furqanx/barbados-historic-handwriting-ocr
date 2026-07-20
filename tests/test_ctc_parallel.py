import pytest

torch = pytest.importorskip("torch")

from src.ctc.parallel import CTCBatchFirstWrapper, ctc_logits_to_time_first
from src.ctc.models.crnn import CRNNCTCConfig, CRNNCTCModel
from src.common.torch_utils import unwrap_model


def test_ctc_batch_first_wrapper_returns_batch_first_logits() -> None:
    model = CRNNCTCModel(
        CRNNCTCConfig(
            vocab_size=12,
            rnn_hidden_size=16,
            rnn_layers=1,
            dropout=0.0,
        )
    )
    wrapper = CTCBatchFirstWrapper(model)
    images = torch.rand(2, 3, 64, 128)

    logits = wrapper(images)

    assert logits.shape == (2, 32, 12)
    assert ctc_logits_to_time_first(logits, wrapper).shape == (32, 2, 12)


def test_unwrap_model_recovers_ctc_wrapped_model() -> None:
    model = CRNNCTCModel(CRNNCTCConfig(vocab_size=12))
    wrapper = CTCBatchFirstWrapper(model)

    assert unwrap_model(wrapper) is model
