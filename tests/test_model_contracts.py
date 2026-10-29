"""Dataset-free behavioral checks; these are not historical metric reproduction."""
import io
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import cxr_h3_research as h3
import cxr_h6_adversarial as h6


class ModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(2026)
        self.net = h3.build_model(h3.Config(experiment='H3B'))

    def test_hierarchical_shapes_and_finite_hybrid_backward(self):
        x = torch.rand(2, 1, 128, 128)
        rec, state = self.net(x)
        self.assertEqual(tuple(rec.shape), tuple(x.shape))
        self.assertEqual(tuple(state['mg'].shape), (2, 128))
        self.assertEqual(tuple(state['qm'].shape), (2, 8, 32, 32))
        cfg = h3.Config(experiment='H3B', local_beta_multiplier=0.35)
        terms = h3.loss_terms(rec, x, state, 1.0, 'hybrid', cfg)
        terms['total'].backward()
        self.assertTrue(all(torch.isfinite(v) for v in terms.values()))
        for module in (self.net.g_stats, self.net.posterior_stats, self.net.out):
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))

    def test_prior_generation_never_calls_encoder(self):
        self.net.eval()
        with patch.object(self.net, 'features', side_effect=AssertionError('encoder used')):
            generated = self.net.generate(2, generator=torch.Generator().manual_seed(33))
        self.assertEqual(tuple(generated.shape), (2, 1, 128, 128))
        self.assertTrue(torch.isfinite(generated).all())
        self.assertTrue(((generated >= 0) & (generated <= 1)).all())

    def test_gaussian_kl_known_cases(self):
        zero = torch.zeros(2, 4)
        self.assertEqual(float(h3.kl_diag(zero, zero)), 0.0)
        self.assertAlmostEqual(float(h3.kl_diag(torch.ones_like(zero), zero)), 2.0, places=6)
        mean, logvar = torch.randn(2, 4), torch.randn(2, 4)
        self.assertAlmostEqual(float(h3.kl_diag(mean, logvar, mean, logvar)), 0.0, places=6)

    def test_decoder_step_keeps_encoder_and_priors_bitwise_identical(self):
        trainable = h6.unlock_decoder(self.net)
        frozen = {name: p.detach().clone() for name, p in self.net.named_parameters() if not p.requires_grad}
        decoder_before = self.net.out[2].weight.detach().clone()
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        x = torch.rand(2, 1, 128, 128)
        gm, zl, _ = h6.posterior_latents(self.net, x, stochastic=False)
        loss, _ = h6.reconstruction_loss(self.net.decoder(gm, zl), x)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(decoder_before, self.net.out[2].weight))
        for name, p in self.net.named_parameters():
            if name in frozen:
                self.assertTrue(torch.equal(frozen[name], p), name)
                self.assertIsNone(p.grad)

    def test_weights_roundtrip_preserves_seeded_prior_samples(self):
        self.net.eval()
        expected = h6.fixed_samples(self.net, 2, 99, 'cpu')
        buffer = io.BytesIO()
        torch.save(self.net.state_dict(), buffer)
        buffer.seek(0)
        other = h3.build_model(h3.Config(experiment='H3B')).eval()
        other.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        actual = h6.fixed_samples(other, 2, 99, 'cpu')
        self.assertTrue(torch.equal(expected, actual))

    def test_image_split_is_repeatable_disjoint_and_class_stratified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'chest_xray'
            for split, count in [('train', 10), ('test', 2)]:
                for label in ('NORMAL', 'PNEUMONIA'):
                    folder = root / split / label
                    folder.mkdir(parents=True)
                    for index in range(count):
                        Image.new('L', (32, 32), color=index * 10).save(folder / f'{index}.png')
            cfg = h3.Config(experiment='H3B', data_root=tmp, batch_size=2)
            train, val, _, a = h3.data_loaders(cfg)
            _, _, _, b = h3.data_loaders(cfg)
            self.assertEqual(a, b)
            self.assertEqual((len(train.dataset), len(val.dataset)), (18, 2))
            self.assertFalse(set(a['train_paths']) & set(a['val_paths']))
            self.assertEqual({Path(p).parts[1] for p in a['val_paths']}, {'NORMAL', 'PNEUMONIA'})
            self.assertTrue(all(p.startswith('train') for p in a['train_paths'] + a['val_paths']))


if __name__ == '__main__':
    unittest.main()
