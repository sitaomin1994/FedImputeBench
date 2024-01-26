from typing import Any, List, Dict, Tuple

# third party
import numpy as np
import torch
from torch import nn, optim
import torch.distributions as td
import torch.nn.functional as F

# hyperimpute absolute
from emf.reproduce_utils import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_init(layer: Any) -> None:
    if type(layer) == nn.Linear:
        torch.nn.init.orthogonal_(layer.weight)


class FusionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, activation_fn):
        super(FusionLayer, self).__init__()

        # Check if there are hidden layers to add
        if num_layers > 1:
            layers = [nn.Linear(input_dim, hidden_dim), activation_fn]
            # hidden layers
            for _ in range(num_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), activation_fn]
            layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            layers = [nn.Linear(input_dim, output_dim)]

        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)


class MaskNet(nn.Module):
    def __init__(self, input_dim, output_dim, out_activation_fn):
        super(MaskNet, self).__init__()
        self.mask_layer = nn.Linear(input_dim, output_dim)
        self.out_activation_fn = out_activation_fn

    def forward(self, u):
        return self.out_activation_fn(self.mask_layer(u))


class GaussianDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, out_activation_fn):
        super(GaussianDecoder, self).__init__()
        self.mu_layer = nn.Linear(input_dim, output_dim)
        self.std_layer = nn.Linear(input_dim, output_dim)
        self.out_activation_fn = out_activation_fn

    def forward(self, u):
        mu = self.out_activation_fn(self.mu_layer(u))
        std = F.softplus(self.std_layer(u))
        return mu, std


class GNR(nn.Module):

    def __init__(
            self,
            num_features: int,
            latent_size: int = 1,
            n_hidden: int = 1,
            seed: int = 0,
            activation_fn: Any = None,
            out_activation_fn: Any = None,
            K: int = 20,
    ) -> None:
        super().__init__()
        set_seed(seed)

        self.num_features = num_features
        self.n_hidden = n_hidden  # number of hidden units in (same for all MLPs)
        self.latent_size = latent_size  # dimension of the latent space
        self.K = K  # number of IS during training

        # encoder
        self.encoder = nn.Sequential(
            torch.nn.Linear(num_features, self.n_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(self.n_hidden, self.n_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(
                self.n_hidden, 2 * self.latent_size
            ),  # the encoder will output both the mean and the diagonal covariance
        ).to(DEVICE)

        # fusion layer
        self.fusion_dim = self.n_hidden
        self.fusion_layer = FusionLayer(
            self.latent_size, self.fusion_dim, self.n_hidden, num_layers=1, activation_fn = nn.Tanh()
        )

        # decoder
        self.decoder = GaussianDecoder(self.fusion_dim, self.num_features, out_activation_fn)

        # mask net
        self.mask_net = MaskNet(self.fusion_dim, self.num_features, out_activation_fn)

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
        all_logp_mgivenx = p_mgivenx.log_prob(tiled_mask.reshape([-1, 1])).reshape([self.K * batch_size, self.num_features])

        logpxobsgivenz = torch.sum(all_log_pxgivenz * tiled_mask, 1).reshape([self.K, batch_size])
        logpz = self.p_z.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)
        logpmgivenx = torch.sum(all_logp_mgivenx, 1).reshape([self.K, batch_size])

        neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpmgivenx + logpz - logq, 0))

        return neg_bound, {}

    def impute(self, x: torch.Tensor, mask: torch.Tensor, L: int) -> torch.Tensor:
        batch_size = x.shape[0]
        p = x.shape[1]

        # encoder
        self.encoder.to(DEVICE)
        self.decoder.to(DEVICE)
        self.mask_net.to(DEVICE)
        out_encoder = self.encoder(x)
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
        recon_x_tiled = recon_x_means * (1 - tiled_mask) + data* (tiled_mask)  # (L * batch_size, p)
        mask_recon = self.mask_net(recon_x_tiled).reshape([-1, 1])  # (L * batch_size*p, 1)
        p_mgivenx = td.Bernoulli(logits=mask_recon)  # (L * batch_size*p, 1)

        # loss
        all_log_pxgivenz_flat = torch.distributions.StudentT(
            loc=recon_x_means.reshape([-1, 1]),
            scale=recon_x_scale.reshape([-1, 1]),
            df=recon_x_df.reshape([-1, 1]),
        ).log_prob(data_flat)  # (L * batch_size, 1)
        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([L * batch_size, p])  # (L * batch_size, p)
        all_log_p_mgivenx = p_mgivenx.log_prob(tiled_mask.reshape([-1, 1])).reshape([L * batch_size, p]) # (L * batch_size, p)

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
            logpxobsgivenz + logpmgivenx + logpz - logq, 0
        )  # these are w_1,....,w_L for all observations in the batch
        xms = xgivenz.sample().reshape([L, batch_size, p])
        xm = torch.einsum("ki,kij->ij", imp_weights, xms)

        # merge imputed values with observed values
        xhat = torch.clone(x)
        xhat[~mask.bool()] = xm[~mask.bool()]

        return xhat
