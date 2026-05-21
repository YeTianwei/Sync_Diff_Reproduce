import unittest

import torch

from losses.fmap_loss import SynchronousDiffusionLoss


class SynchronousDiffusionLossTest(unittest.TestCase):
    @staticmethod
    def _basis(num_verts, num_evecs):
        q, _ = torch.linalg.qr(torch.randn(num_verts, num_evecs))
        return q.unsqueeze(0)

    def test_loss_is_finite_scalar_and_backpropagates(self):
        torch.manual_seed(0)
        batch_size, num_x, num_y, num_evecs = 1, 7, 5, 4

        evecs_x = self._basis(num_x, num_evecs)
        evecs_y = self._basis(num_y, num_evecs)
        evecs_trans_x = evecs_x.transpose(1, 2).contiguous()
        evecs_trans_y = evecs_y.transpose(1, 2).contiguous()
        evals_x = torch.linspace(0.0, 3.0, num_evecs).unsqueeze(0)
        evals_y = torch.linspace(0.0, 2.5, num_evecs).unsqueeze(0)

        pxy_logits = torch.randn(batch_size, num_x, num_y, requires_grad=True)
        pyx_logits = torch.randn(batch_size, num_y, num_x, requires_grad=True)
        pxy = torch.softmax(pxy_logits, dim=-1)
        pyx = torch.softmax(pyx_logits, dim=-1)

        loss_fn = SynchronousDiffusionLoss(num_random=8, max_time=1e-2)
        loss = loss_fn(pxy, pyx, evals_x, evals_y, evecs_x, evecs_y,
                       evecs_trans_x, evecs_trans_y)
        loss.backward()

        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.isfinite(pxy_logits.grad).all().item())
        self.assertTrue(torch.isfinite(pyx_logits.grad).all().item())


if __name__ == '__main__':
    unittest.main()
