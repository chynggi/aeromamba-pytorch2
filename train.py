"""
This code is based on Facebook's HDemucs code: https://github.com/facebookresearch/demucs
"""
import itertools
import logging
import os
import numpy as np
import hydra
import wandb
import random

from src.ddp.executor import start_ddp_workers
from src.models import modelFactory
from src.utils import print_network
from src.wandb_logger import _init_wandb_run

logger = logging.getLogger(__name__)


def run(args):
    import torch

    from src.ddp import distrib
    from src.data.datasets import LrHrSet
    from src.solver import Solver
    logger.info(f'calling distrib.init')
    distrib.init(args)
    # Note: set_detect_anomaly is now optional and can be enabled via config for debugging
    # For production, it's disabled for better performance
    if hasattr(args, 'detect_anomaly') and args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    _init_wandb_run(args)

    if distrib.rank == 0:
        if not os.path.exists(args.samples_dir):
            os.makedirs(args.samples_dir)

    # torch also initialize cuda seed if available
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Enable TF32 for better performance on Ampere GPUs and later (CUDA 12.x compatible)
    if torch.cuda.is_available() and hasattr(args, 'use_tf32') and args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled for improved performance on Ampere+ GPUs")
    torch.backends.cudnn.benchmark = True
    
    models = modelFactory.get_model(args)
    for model_name, model in models.items():
        print_network(model_name, model, logger)
    wandb.watch(tuple(models.values()), log=args.wandb.log, log_freq=args.wandb.log_freq)

    if args.show:
        logger.info(models)
        mb = sum(p.numel() for p in models.parameters()) * 4 / 2 ** 20
        logger.info('Size: %.1f MB', mb)
        return

    assert args.experiment.batch_size % distrib.world_size == 0
    args.experiment.batch_size //= distrib.world_size

    # Building datasets and loaders
    tr_dataset = LrHrSet(args.dset.train, args.experiment.lr_sr, args.experiment.hr_sr,
                         args.experiment.stride, args.experiment.segment, upsample=args.experiment.upsample, fixed_n_examples=args.experiment.fixed_n_examples)
    
    # Filter items based on the threshold (for silent and low-power segments). Threshold value can be tuned
    if args.experiment.power_threshold > 0:
        filtered_lr_set = []
        filtered_hr_set = []
        for i in range(len(tr_dataset.lr_set)):
            lr_signal = tr_dataset.lr_set[i]
            hr_signal = tr_dataset.hr_set[i]
            if hr_signal is not None:
                hr_power = torch.square(hr_signal).sum() / args.experiment.hr_sr  
                if hr_power >= args.experiment.power_threshold:
                    filtered_lr_set.append(lr_signal)
                    filtered_hr_set.append(hr_signal)

        # Replace lr_set and hr_set with filtered lists
        tr_dataset.lr_set = filtered_lr_set
        tr_dataset.hr_set = filtered_hr_set

    tr_loader = distrib.loader(tr_dataset, batch_size=args.experiment.batch_size, shuffle=True,
                               num_workers=args.num_workers)

    if args.dset.valid:
        args.valid_equals_test = args.dset.valid == args.dset.test

    # Validation and Test batch size, segments and strides can be set differently by the user depending on GPU resources

    if args.dset.valid:
        cv_dataset = LrHrSet(args.dset.valid, args.experiment.lr_sr, args.experiment.hr_sr,
                            args.experiment.stride, args.experiment.segment, upsample=args.experiment.upsample, fixed_n_examples=args.experiment.fixed_n_examples)
        cv_loader = distrib.loader(cv_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    else:
        cv_loader = None

    if args.dset.test:
        tt_dataset = LrHrSet(args.dset.test, args.experiment.lr_sr, args.experiment.hr_sr,
                             stride=40, segment=10, with_path=True, upsample=args.experiment.upsample)
        tt_loader = distrib.loader(tt_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    else:
        tt_loader = None
    data = {"tr_loader": tr_loader, "cv_loader": cv_loader, "tt_loader": tt_loader}

    if torch.cuda.is_available() and args.device=='cuda':
        for model in models.values():
            model.cuda()

    # optimizer
    if args.optim == "adam":
        optimizer = torch.optim.Adam(models['generator'].parameters(), lr=args.lr, betas=(0.9, args.beta2))
    else:
        logger.fatal('Invalid optimizer %s', args.optim)
        os._exit(1)

    optimizers = {'optimizer': optimizer}


    if 'adversarial' in args.experiment and args.experiment.adversarial:
        disc_optimizer = torch.optim.Adam(
            itertools.chain(*[models[disc_name].parameters() for disc_name in
                              args.experiment.discriminator_models]),
            args.lr, betas=(0.9, args.beta2))
        optimizers.update({'disc_optimizer': disc_optimizer})


    # Construct Solver
    solver = Solver(data, models, optimizers, args)
    solver.train()

    distrib.close()



def _main(args):
    global __file__
    print(args)
    # Updating paths in config
    for key, value in args.dset.items():
        if isinstance(value, str):
            args.dset[key] = hydra.utils.to_absolute_path(value)
    __file__ = hydra.utils.to_absolute_path(__file__)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("src").setLevel(logging.DEBUG)

    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)



    if args.ddp and args.rank is None:
        start_ddp_workers(args)
    else:
        run(args)

    wandb.finish()


@hydra.main(config_path="conf", config_name="main_config")  # for latest version of hydra=1.0
def main(args):
    try:
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        # Hydra intercepts exit code, fixed in beta but I could not get the beta to work
        os._exit(1)


if __name__ == "__main__":
    main()
