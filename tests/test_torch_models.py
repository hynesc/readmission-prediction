import unittest


try:
    import torch

    from readmission.config import ModelConfig
    from readmission.models.content import CONTENTModel
    from readmission.models.content_original import OriginalStyleCONTENTModel
    from readmission.models.transformer import CausalTransformerReadmissionModel
    from readmission.train import masked_bce_loss, model_loss
except Exception:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TorchModelTests(unittest.TestCase):
    def test_masked_loss_ignores_padded_visits(self):
        logits = torch.tensor([[0.0, 10.0]])
        labels_a = torch.tensor([[0.0, 0.0]])
        labels_b = torch.tensor([[0.0, 1.0]])
        mask = torch.tensor([[True, False]])
        self.assertAlmostEqual(float(masked_bce_loss(logits, labels_a, mask)), float(masked_bce_loss(logits, labels_b, mask)))

    def test_content_topic_summary_ignores_extra_padding(self):
        config = ModelConfig(model_type="content", visit_dim=4, hidden_dim=4, code_embedding_dim=4, num_topics=3)
        model = CONTENTModel(vocab={str(i): i for i in range(8)}, config=config)
        visits = torch.randn(1, 2, 4)
        mask = torch.tensor([[True, True]])
        padded = torch.cat([visits, torch.randn(1, 2, 4) * 100], dim=1)
        padded_mask = torch.tensor([[True, True, False, False]])
        self.assertTrue(torch.allclose(model.topic_summary(visits, mask), model.topic_summary(padded, padded_mask), atol=1e-6))

    def test_transformer_causal_mask_blocks_future_positions(self):
        mask = CausalTransformerReadmissionModel.causal_mask(4, torch.device("cpu"))
        self.assertFalse(bool(mask[0, 0]))
        self.assertTrue(bool(mask[0, 3]))
        self.assertFalse(bool(mask[3, 0]))

    def test_original_style_content_exposes_kl_regularization(self):
        config = ModelConfig(
            model_type="content_original",
            embedding_type="multi_hot",
            visit_dim=4,
            hidden_dim=4,
            num_topics=3,
            kl_weight=0.5,
        )
        model = OriginalStyleCONTENTModel(vocab={str(i): i for i in range(8)}, config=config)
        codes = torch.tensor([[[2, 3], [4, 0], [0, 0]]])
        code_mask = torch.tensor([[[True, True], [True, False], [False, False]]])
        visit_mask = torch.tensor([[True, True, False]])
        labels = torch.tensor([[0.0, 1.0, 0.0]])
        logits = model(codes, code_mask, visit_mask)
        self.assertEqual(tuple(logits.shape), (1, 3))
        self.assertIsNotNone(model.last_kl_loss)
        self.assertGreaterEqual(float(model.regularization_loss()), 0.0)
        self.assertGreaterEqual(float(model_loss(model, logits, labels, visit_mask)), 0.0)


if __name__ == "__main__":
    unittest.main()
