import unittest

import torch

from networks.fmap_network import FasterRegularizedFMNet, RegularizedFMNet


class FasterRegularizedFMNetTest(unittest.TestCase):
    @staticmethod
    def _basis(num_verts, num_evecs):
        q, _ = torch.linalg.qr(torch.randn(num_verts, num_evecs))
        return q.unsqueeze(0)

    def test_matches_loop_solver(self):
        torch.manual_seed(0)
        batch_size, num_x, num_y, num_evecs, num_feats = 2, 17, 19, 8, 11

        feat_x = torch.randn(batch_size, num_x, num_feats)
        feat_y = torch.randn(batch_size, num_y, num_feats)
        evecs_x = self._basis(num_x, num_evecs).repeat(batch_size, 1, 1)
        evecs_y = self._basis(num_y, num_evecs).repeat(batch_size, 1, 1)
        evecs_trans_x = evecs_x.transpose(1, 2).contiguous()
        evecs_trans_y = evecs_y.transpose(1, 2).contiguous()
        evals_x = torch.linspace(0.01, 4.0, num_evecs).unsqueeze(0).repeat(batch_size, 1)
        evals_y = torch.linspace(0.02, 4.5, num_evecs).unsqueeze(0).repeat(batch_size, 1)

        loop_solver = RegularizedFMNet(bidirectional=True)
        fast_solver = FasterRegularizedFMNet(bidirectional=True)

        Cxy_loop, Cyx_loop = loop_solver(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)
        Cxy_fast, Cyx_fast = fast_solver(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)

        self.assertTrue(torch.allclose(Cxy_loop, Cxy_fast, rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.allclose(Cyx_loop, Cyx_fast, rtol=1e-5, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
