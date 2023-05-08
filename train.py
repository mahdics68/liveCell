import torch
import os
import torch_em
from torch_em.model import UNet2d
from torch_em.data.datasets import get_livecell_loader
from torch_em.util.debug import check_loader
from torch_em.trainer.tensorboard_logger import TensorboardLogger

def train_boundaries(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    n_out = 2
    model = UNet2d(in_channels=1, out_channels=n_out, initial_features=64,
                   final_activation="Sigmoid")
    model.to(device)

    patch_shape = (256, 256)
    train_loader = get_livecell_loader(
        args.input, patch_shape, "train",
        download=True, boundaries=True, batch_size=args.batch_size, cell_types = [args.cell_types]
    )
    val_loader = get_livecell_loader(
        args.input, patch_shape, "val",
        download=True, boundaries=True, batch_size=1,cell_types = [args.cell_types]
    )
    
    loss = torch_em.loss.DiceLoss()

    trainer = torch_em.default_segmentation_trainer(
        name="livecell-boundary-model",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=loss,
        learning_rate=1e-4,
        device=device,
        mixed_precision=True,
        log_image_interval=50,
        save_root = "/scratch/users/menayat/models/livecell-unet-A172"
    )
    trainer.fit(iterations=args.n_iterations)


if __name__ == '__main__':
    parser = torch_em.util.parser_helper(
        default_batch_size=2
    )
    parser.add_argument("--cell_types", type=str, default=None)
    args = parser.parse_args()
    train_boundaries(args)
