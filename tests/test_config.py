import unittest

from readmission.config import load_config


class ConfigTests(unittest.TestCase):
    def test_original_style_content_config_is_explicit(self):
        config = load_config("configs/content_original.yaml")
        self.assertEqual(config.model.model_type, "content_original")
        self.assertEqual(config.model.embedding_type, "multi_hot")
        self.assertEqual(config.model.hidden_dim, 200)
        self.assertEqual(config.model.num_topics, 50)


if __name__ == "__main__":
    unittest.main()
