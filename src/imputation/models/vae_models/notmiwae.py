# stdlib
from typing import Any, List, Dict, Tuple

# third party
import numpy as np
import torch
from torch import nn, optim
import torch.distributions as td

# hyperimpute absolute
from emf.reproduce_utils import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_init(layer: Any) -> None:
    if type(layer) == nn.Linear:
        torch.nn.init.orthogonal_(layer.weight)


class MaskNet(nn.Module):
    """
    This implements the Mask net that is used in notmiwae's implementation for self-masking mechanism
    """

    def __init__(self, input_dim: int):
        """
        Args:
            input_dim: Dimension of observed features.
            device: torch device to use.
        """
        super().__init__()
        self._device = DEVICE
        self.__input_dim = input_dim

        self.W = torch.nn.Parameter(torch.zeros([1, input_dim], device=DEVICE), requires_grad=True)
        self.b = torch.nn.Parameter(torch.zeros([1, input_dim], device=DEVICE), requires_grad=True)
        self._device = DEVICE

    def forward(self, x):
        """
        Args:
            x: Input tensor with shape (batch_size, input_dim).

        Returns:
            encoded: Encoded output tensor with shape (batch_size, input_dim)
        """  # Run masked values through model.
        output = -self.W * (x - self.b)
        return output


class MaskNet2(nn.Module):
    """
    This implements the Mask net that is used in notmiwae's implementation for self-masking mechanism
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        """
        Args:
            input_dim: Dimension of observed features.
            device: torch device to use.
        """
        super().__init__()
        self._device = DEVICE
        self.__input_dim = input_dim

        self.decoder = nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, input_dim),
        ).to(DEVICE)
        self._device = DEVICE

    def forward(self, x):
        """
        Args:
            x: Input tensor with shape (batch_size, input_dim).

        Returns:
            encoded: Encoded output tensor with shape (batch_size, input_dim)
        """  # Run masked values through model.
        output = self.decoder(x)
        return output


class NOTMIWAE(nn.Module):
    """
    Not-MIWAE imputation plugin

    Args:
        n_epochs: int
            Number of training iterations
        batch_size: int
            Batch size
        latent_size: int
            dimension of the latent space
        n_hidden: int
            number of hidden units
        K: int
            number of IS during training
        random_state: int
            random seed

    Reference: "MIWAE: Deep Generative Modelling and Imputation of Incomplete Data", Pierre-Alexandre Mattei,
    Jes Frellsen
    Original code: https://github.com/pamattei/miwae
    """

    def __init__(
            self,
            num_features: int,
            latent_size: int = 1,
            n_hidden: int = 1,
            seed: int = 0,
            K: int = 20,
    ) -> None:
        super().__init__()
        set_seed(seed)

        self.num_features = num_features
        self.n_hidden = n_hidden  # number of hidden units in (same for all MLPs)
        self.latent_size = latent_size  # dimension of the latent space
        self.K = K  # number of IS during training

        # mask encoder
        self.mask_enc_dim = 0
        self.mask_enc = nn.Sequential(
            torch.nn.Linear(num_features, self.n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_hidden, self.mask_enc_dim)
        )

        # encoder
        self.encoder = nn.Sequential(
            torch.nn.Linear(num_features + self.mask_enc_dim, self.n_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(self.n_hidden, self.n_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(
                self.n_hidden, 2 * self.latent_size
            ),  # the encoder will output both the mean and the diagonal covariance
        ).to(DEVICE)

        # decoder
        self.decoder = nn.Sequential(
            torch.nn.Linear(self.latent_size, self.n_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(self.n_hidden, self.n_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(
                self.n_hidden, 3 * num_features
            ),
            # the decoder will output both the mean, the scale, and the number of degrees of freedoms (hence the 3*p)
        ).to(DEVICE)

        self.mask_net = MaskNet(num_features).to(DEVICE)

        self.p_z = td.Independent(
            td.Normal(loc=torch.zeros(self.latent_size).to(DEVICE), scale=torch.ones(self.latent_size).to(DEVICE)), 1
        )

    @staticmethod
    def name() -> str:
        return "miwae"

    def init(self):
        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)

    def compute_loss(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        x, mask = inputs  # x - data, mask - missing mask
        batch_size = x.shape[0]

        # encoder
        mask_ = self.mask_enc(mask)
        x_ = torch.concat([x, mask_], dim=1)
        out_encoder = self.encoder(x)
        mu, logvar = out_encoder[..., :self.latent_size], out_encoder[..., self.latent_size:(2 * self.latent_size)]

        q_zgivenxobs = td.Independent(td.Normal(loc=mu, scale=torch.nn.Softplus()(logvar)), 1)
        zgivenx = q_zgivenxobs.rsample([self.K])
        zgivenx_flat = zgivenx.reshape([self.K * batch_size, self.latent_size])

        # decoder
        out_decoder = self.decoder(zgivenx_flat)
        recon_x_means = out_decoder[..., :self.num_features]
        recon_x_scale = torch.nn.Softplus()(out_decoder[..., self.num_features:(2 * self.num_features)]) + 0.001
        recon_x_degree_freedom = torch.nn.Softplus()(out_decoder[..., (2 * self.num_features):]) + 3

        # compute loss
        data = torch.Tensor.repeat(x, [self.K, 1]).to(DEVICE)
        data_flat = torch.Tensor.repeat(x, [self.K, 1]).reshape([-1, 1]).to(DEVICE)
        tiled_mask = torch.Tensor.repeat(mask, [self.K, 1]).to(DEVICE)

        # mask net
        recon_x_tiled = recon_x_means * (1 - tiled_mask) + data * tiled_mask  # (K * batch_size, p)
        mask_recon = self.mask_net(recon_x_tiled).to(DEVICE).reshape([-1, 1])  # (K * batch_size*p, 1)
        p_mgivenx = td.Bernoulli(logits=mask_recon)  # (K * batch_size*p, 1)

        # compute loss
        all_log_pxgivenz_flat = torch.distributions.StudentT(
            loc=recon_x_means.reshape([-1, 1]),
            scale=recon_x_scale.reshape([-1, 1]),
            df=recon_x_degree_freedom.reshape([-1, 1]),
        ).log_prob(data_flat)
        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([self.K * batch_size, self.num_features])
        all_logp_mgivenx = p_mgivenx.log_prob(tiled_mask.reshape([-1, 1])).reshape(
            [self.K * batch_size, self.num_features])

        logpxobsgivenz = torch.sum(all_log_pxgivenz * tiled_mask, 1).reshape([self.K, batch_size])
        logpz = self.p_z.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)
        logpmgivenx = torch.sum(all_logp_mgivenx, 1).reshape([self.K, batch_size])

        neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + 5*logpmgivenx + logpz - logq, 0))

        return neg_bound, {}

    def impute(self, x: torch.Tensor, mask: torch.Tensor, L: int) -> torch.Tensor:
        batch_size = x.shape[0]
        p = x.shape[1]

        # encoder
        self.encoder.to(DEVICE)
        self.decoder.to(DEVICE)
        self.mask_net.to(DEVICE)
        self.mask_enc.to(DEVICE)
        mask_ = self.mask_enc(mask)
        x_ = torch.concat([x, mask_], dim=1)
        out_encoder = self.encoder(x)
        # out_encoder = self.encoder(x_)
        mu = out_encoder[..., : self.latent_size]  # (batch_size, latent_size)
        logvar = torch.nn.Softplus()(
            out_encoder[..., self.latent_size: (2 * self.latent_size)])  # (batch_size, latent_size)
        q_zgivenxobs = td.Independent(td.Normal(loc=mu, scale=logvar), 1)

        zgivenx = q_zgivenxobs.rsample([L])  # (L, batch_size, latent_size)
        zgivenx_flat = zgivenx.reshape([L * batch_size, self.latent_size])  # (L * batch_size, latent_size)

        # decoder
        out_decoder = self.decoder(zgivenx_flat)
        recon_x_means = out_decoder[..., :p]  # (L*batch_size, p)
        recon_x_scale = torch.nn.Softplus()(out_decoder[..., p: (2 * p)]) + 0.001  # (L*batch_size, p)
        recon_x_df = torch.nn.Softplus()(out_decoder[..., (2 * p): (3 * p)]) + 3  # (L*batch_size, p)

        # ground truth data
        data = torch.Tensor.repeat(x, [L, 1]).to(DEVICE)  # (L * batch_size, p)
        data_flat = torch.Tensor.repeat(x, [L, 1]).reshape([-1, 1]).to(DEVICE)  # (L * batch_size*p, 1)
        tiled_mask = torch.Tensor.repeat(mask, [L, 1]).to(DEVICE)  # (L * batch_size, p)

        # mask net
        recon_x_tiled = recon_x_means * (1 - tiled_mask) + data * (tiled_mask)  # (L * batch_size, p)
        mask_recon = self.mask_net(recon_x_tiled).reshape([-1, 1])  # (L * batch_size*p, 1)
        p_mgivenx = td.Bernoulli(logits=mask_recon)  # (L * batch_size*p, 1)

        # loss
        all_log_pxgivenz_flat = torch.distributions.StudentT(
            loc=recon_x_means.reshape([-1, 1]),
            scale=recon_x_scale.reshape([-1, 1]),
            df=recon_x_df.reshape([-1, 1]),
        ).log_prob(data_flat)  # (L * batch_size, 1)
        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([L * batch_size, p])  # (L * batch_size, p)
        all_log_p_mgivenx = p_mgivenx.log_prob(tiled_mask.reshape([-1, 1])).reshape(
            [L * batch_size, p])  # (L * batch_size, p)

        logpxobsgivenz = torch.sum(all_log_pxgivenz * tiled_mask, 1).reshape([L, batch_size])  # (L, batch_size)
        logpz = self.p_z.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)
        logpmgivenx = torch.sum(all_log_p_mgivenx, 1).reshape([L, batch_size])

        # imputation weighted samples
        xgivenz = td.Independent(
            td.StudentT(
                loc=recon_x_means,
                scale=recon_x_scale,
                df=recon_x_df,
            ),
            1
        )

        imp_weights = torch.nn.functional.softmax(
            logpxobsgivenz + 5*logpmgivenx + logpz - logq, 0
        )  # these are w_1,....,w_L for all observations in the batch
        xms = xgivenz.sample().reshape([L, batch_size, p])
        xm = torch.einsum("ki,kij->ij", imp_weights, xms)

        # merge imputed values with observed values
        xhat = torch.clone(x)
        xhat[~mask.bool()] = xm[~mask.bool()]

        return xhat
