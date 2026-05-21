import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class SquaredFrobeniusLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, a, b):
        loss = torch.sum(torch.abs(a - b) ** 2, dim=(-2, -1))
        return self.loss_weight * torch.mean(loss)


@LOSS_REGISTRY.register()
class SynchronousDiffusionLoss(nn.Module):
    """
    Multiscale synchronous diffusion regularisation.

    The pointwise maps are expected to follow the convention used in FMNetModel:
    Pyx transfers functions from x to y, and Pxy transfers functions from y to x.
    """

    def __init__(self, loss_weight=1.0, num_random=128, max_time=1e-2, bidirectional=True):
        super().__init__()
        self.loss_weight = loss_weight
        self.num_random = num_random
        self.max_time = max_time
        self.bidirectional = bidirectional

    @staticmethod
    def _diffuse(feats, evals, evecs, evecs_trans, times):
        coeffs = torch.bmm(evecs_trans, feats)
        scales = torch.exp(-evals.unsqueeze(-1) * times.unsqueeze(1))
        return torch.bmm(evecs, coeffs * scales)

    def _single_direction(self, P_src_tgt, P_tgt_src,
                          evals_src, evals_tgt,
                          evecs_src, evecs_tgt,
                          evecs_trans_src, evecs_trans_tgt):
        batch_size, num_src, _ = evecs_src.shape
        funcs_src = torch.randn(batch_size, num_src, self.num_random,
                                device=evecs_src.device, dtype=evecs_src.dtype)
        funcs_src = F.normalize(funcs_src, p=2, dim=-1)
        times = torch.rand(batch_size, self.num_random,
                           device=evecs_src.device, dtype=evecs_src.dtype) * self.max_time

        funcs_tgt = torch.bmm(P_src_tgt, funcs_src)
        diffuse_src = self._diffuse(funcs_src, evals_src, evecs_src, evecs_trans_src, times)
        diffuse_tgt = self._diffuse(funcs_tgt, evals_tgt, evecs_tgt, evecs_trans_tgt, times)
        diffuse_tgt_src = torch.bmm(P_tgt_src, diffuse_tgt)

        loss = torch.sum((diffuse_src - diffuse_tgt_src) ** 2, dim=(-2, -1))
        return torch.mean(loss)

    def forward(self, Pxy, Pyx, evals_x, evals_y, evecs_x, evecs_y, evecs_trans_x, evecs_trans_y):
        loss = self._single_direction(Pyx, Pxy, evals_x, evals_y, evecs_x, evecs_y,
                                      evecs_trans_x, evecs_trans_y)
        if self.bidirectional:
            loss += self._single_direction(Pxy, Pyx, evals_y, evals_x, evecs_y, evecs_x,
                                           evecs_trans_y, evecs_trans_x)
        return self.loss_weight * loss


@LOSS_REGISTRY.register()
class SURFMNetLoss(nn.Module):
    """
    Loss as presented in the SURFMNet paper.
    Orthogonality + Bijectivity + Laplacian Commutativity
    """

    def __init__(self, w_bij=1.0, w_orth=1.0, w_lap=1e-3):
        """
        Init SURFMNetLoss

        Args:
            w_bij (float, optional): Bijectivity penalty weight. Default 1e3.
            w_orth (float, optional): Orthogonality penalty weight. Default 1e3.
            w_lap (float, optional): Laplacian commutativity penalty weight. Default 1.0.
        """
        super(SURFMNetLoss, self).__init__()
        assert w_bij >= 0 and w_orth >= 0 and w_lap >= 0
        self.w_bij = w_bij
        self.w_orth = w_orth
        self.w_lap = w_lap

    def forward(self, C12, C21, evals_1, evals_2):
        """
        Compute bijectivity loss + orthogonality loss
                            + Laplacian commutativity loss
                            + descriptor preservation via commutativity loss

        Args:
            C12 (torch.Tensor): matrix representation of functional map (1->2). Shape: [N, K, K]
            C21 (torch.Tensor): matrix representation of functional map (2->1). Shape: [N, K, K]
            evals_1 (torch.Tensor): eigenvalues of shape 1. Shape [N, K]
            evals_2 (torch.Tensor): eigenvalues of shape 2. Shape [N, K]
        """
        criterion = SquaredFrobeniusLoss()
        eye = torch.eye(C12.shape[1], C12.shape[2], device=C12.device).unsqueeze(0)
        eye_batch = torch.repeat_interleave(eye, repeats=C12.shape[0], dim=0)

        losses = dict()
        # Bijectivity penalty
        if self.w_bij > 0:
            bijectivity_loss = criterion(torch.bmm(C12, C21), eye_batch) + criterion(torch.bmm(C21, C12), eye_batch)
            bijectivity_loss *= self.w_bij
            losses['l_bij'] = bijectivity_loss

        # Orthogonality penalty
        if self.w_orth > 0:
            orthogonality_loss = criterion(torch.bmm(C12.transpose(1, 2), C12), eye_batch) + \
                                 criterion(torch.bmm(C21.transpose(1, 2), C21), eye_batch)
            orthogonality_loss *= self.w_orth
            losses['l_orth'] = orthogonality_loss

        # Laplacian commutativity penalty
        if self.w_lap > 0:
            laplacian_loss = criterion(torch.einsum('abc,ac->abc', C12, evals_1),
                                       torch.einsum('ab,abc->abc', evals_2, C12))
            laplacian_loss += criterion(torch.einsum('abc,ac->abc', C21, evals_2),
                                        torch.einsum('ab,abc->abc', evals_1, C21))
            laplacian_loss *= self.w_lap
            losses['l_lap'] = laplacian_loss

        return losses


@LOSS_REGISTRY.register()
class PartialFmapsLoss(nn.Module):
    def __init__(self, w_bij=1.0, w_orth=1.0):
        """
        Init PartialFmapsLoss
        Args:
            w_bij (float, optional): Bijectivity penalty weight. Default 1.0.
            w_orth (float, optional): Orthogonality penalty weight. Default 1.0.
        """
        super(PartialFmapsLoss, self).__init__()
        assert w_bij >= 0 and w_orth >= 0, 'Loss weight should be non-negative.'
        self.w_bij = w_bij
        self.w_orth = w_orth

    def forward(self, C_fp, C_pf, evals_full, evals_partial):
        assert C_fp.shape[0] == 1, 'Currently, only support batch size = 1'
        criterion = SquaredFrobeniusLoss()
        C_fp, C_pf = C_fp[0], C_pf[0]
        evals_full, evals_partial = evals_full[0], evals_partial[0]

        # compute area ratio between full shape and partial shape r
        r = min((evals_partial < evals_full.max()).sum(), C_fp.shape[0] - 1)
        eye = torch.zeros_like(C_fp)
        eye[torch.arange(0, r + 1), torch.arange(0, r + 1)] = 1.0

        if self.w_bij > 0:
            bijectivity_loss = self.w_bij * criterion(torch.matmul(C_fp, C_pf), eye)
        else:
            bijectivity_loss = 0.0

        if self.w_orth > 0:
            orthogonality_loss = self.w_bij * criterion(torch.matmul(C_fp, C_fp.t()), eye)
        else:
            orthogonality_loss = 0.0

        return {'l_bij': bijectivity_loss, 'l_orth': orthogonality_loss}
