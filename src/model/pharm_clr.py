from flash.core.optimizers import LARS, LinearWarmupCosineAnnealingLR
from lightning import LightningModule
import torch

# from utils import visualize_pharm
from dataset import AugmentationModule
from .projector import Projection
from .encoder import Encoder


class PharmCLR(LightningModule):
    def __init__(self, **params):
        super(PharmCLR, self).__init__()
        self.save_hyperparameters()

        # SimCLR architecture
        self.transform = AugmentationModule(train=True)
        self.val_transform = AugmentationModule(train=False)
        self.encoder = Encoder(
            input_dim=self.hparams.num_node_features,
            hidden_dim=self.hparams.output_dims_conv,
            output_dim=self.hparams.output_dims_lin,
            n_conv_layers=self.hparams.n_layers_conv,
            num_edge_features=self.hparams.num_edge_features,
            dropout=self.hparams.dropout,
        )
        input_dimension = self.hparams.output_dims_lin
        self.projection_head = Projection(input_dimension, input_dimension, 64)

        # validation embeddings
        self.val_embeddings = []

    def forward(self, data):
        # Optional: Visualization of the input pharmacophore
        # visualize_pharm(
        #     [data[0].clone(), self.transform(data[0].clone()), self.transform(data[0].clone())]
        # )

        representation = self.encoder(data)
        embedding = self.projection_head(representation)

        return embedding

    def setup(self, stage):
        global_batch_size = self.trainer.world_size * self.hparams.batch_size
        self.train_iters_per_epoch = self.hparams.num_samples // global_batch_size

    def exclude_from_wt_decay(
        self, named_params, weight_decay, skip_list=["bias", "bn"]
    ):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {"params": excluded_params, "weight_decay": 0.0},
        ]

    def configure_optimizers(self):
        # TRICK 1 (Use lars + filter weights)
        # exclude certain parameters
        parameters = self.exclude_from_wt_decay(
            self.named_parameters(), weight_decay=self.hparams.opt_weight_decay
        )

        optimizer = LARS(
            parameters, lr=self.hparams.learning_rate, momentum=self.hparams.opt_eta
        )

        # Trick 2 (after each step)
        warmup_epochs = self.hparams.warmup_epochs * self.train_iters_per_epoch
        max_epochs = self.trainer.max_epochs * self.train_iters_per_epoch

        linear_warmup_cosine_decay = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            warmup_start_lr=0,
            eta_min=0,
        )

        scheduler = {
            "scheduler": linear_warmup_cosine_decay,
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def nt_xent_loss(self, out_1, out_2):
        out = torch.cat([out_1, out_2], dim=0)
        n_samples = len(out)

        # Full similarity matrix
        cov = torch.mm(out, out.t().contiguous())
        sim = torch.exp(cov / self.hparams.temperature)

        mask = ~torch.eye(n_samples, device=sim.device).bool()
        neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

        # Positive similarity
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.hparams.temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / neg).mean()
        return loss

    def shared_step(self, batch, batch_idx):
        out1 = self(self.transform(batch))
        out2 = self(self.transform(batch))

        return self.nt_xent_loss(out1, out2)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log(
            "hp/train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=len(batch),
        )

        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_embeddings = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # val loss calculation
        if dataloader_idx == 0:
            val_loss = self.shared_step(batch, batch_idx)
            self.log(
                "hp/val_loss",
                val_loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=len(batch),
            )

            # rankme criterion
            self.val_embeddings.append(self(self.val_transform(batch)))

            return val_loss

    def on_validation_epoch_end(self) -> None:
        epsilon = 1e-7
        embeddings = torch.vstack(self.val_embeddings)
        _, singular_values, _ = torch.svd(embeddings)
        singular_values /= torch.sum(singular_values)
        singular_values += epsilon
        rank_me = torch.exp(-torch.sum(singular_values * torch.log(singular_values)))
        self.log(
            "hp/rank_me",
            rank_me,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def on_train_start(self):
        self.logger.log_hyperparams(
            self.hparams,
            {"hp/rank_me": 0},
        )

    def predict_step(self, batch, batch_idx):
        if "mol_id" in batch.keys:
            return self(self.val_transform(batch)), batch.mol_id
        else:
            return self(self.val_transform(batch))
