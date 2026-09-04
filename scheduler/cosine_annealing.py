from torch.optim.lr_scheduler import CosineAnnealingLR as _CosineAnnealingLR


class CosineAnnealingLR(_CosineAnnealingLR):
    NAME = "cosine_annealing"
    """
    Wrapper of PyTorch CosineAnnealingLR.
    """
    def __init__(
        self,
        optimizer,
        args
    ):
        T_max = getattr(args, "T_max", 100)
        eta_min = getattr(args, "eta_min",1e-5)
        last_epoch = getattr(args, "last_epoch",-1)

        super().__init__(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
            last_epoch=last_epoch
        )