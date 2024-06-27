from torch import optim
import argparse


def create_parser(desctription=""):
    """Function to handle input arguments"""
    parser = argparse.ArgumentParser(description=desctription)

    #------------ Incremental learning scenario
    scenario_choices = ["task", "class"]
    parser.add_argument("--scenario", type=str, default="task", choices=scenario_choices)
    parser.add_argument("--tasks", type=int, default=10, help="Number of tasks")

    parser.add_argument("--classes_per_task", type=int, default=5, help="Number of classes in a task")

    #---------- Device used to train a network - GPU or CPU
    device_choices = ["cpu", "gpu"]
    parser.add_argument("--device", type=str, default="gpu", choices=device_choices)

    parser.add_argument("--multitask_dataloading", action="store_true", default=True)

    # Number of epochs to train single task
    parser.add_argument("--epochs", type=int, default=45)

    # Optimizer
    optim_choices = ["Adam", "SGD", "RMSprop"]
    parser.add_argument("--optimizer", type=str, default="SGD", choices=optim_choices)

    # Hyperparameters for optimizers
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.99, help="smoothing constant for RMSprop optimizer")
    parser.add_argument(
        "--betas",
        nargs="+",
        type=float,
        default=(0.9, 0.999),
        help="coefficients for computing running averages of gradient and its square in ADAM optimizer",
    )

    # Batch sizes
    parser.add_argument("--train_bs", type=int, default=10)
    parser.add_argument("--test_bs", type=int, default=14)

    # EWC params
    parser.add_argument("--lambda_param", type=float, default=1e3)
    parser.add_argument("--fisher_num_batches", type=int, default=None)

    # PackNet params
    parser.add_argument("--prune_perc", type=float, default=0.5)
    parser.add_argument("--retrain_epochs", type=int, default=25)

    # Flag to decide whether train base model without CL to compare
    parser.add_argument("--base_model", action="store_true", default=True, help="train base model to compare")

    # Flags in script compare_all.py
    parser.add_argument("--ewc", action="store_true", default=False, help="Elastic Weights Consolidation")
    parser.add_argument("--packnet", action="store_true", default=False, help="PackNet algorithm with network pruning")
    # Flag to decide whether to perform joint training to compare
    parser.add_argument("--joint_train", action="store_true", default=False, help="joint training to compare")

    # Flag to print info while program is running
    parser.add_argument("--verbose", action="store_false", default=True, help="print info during program run")
    
    # Flag to save generated plots
    parser.add_argument(
        "--save_plots", action="store_true", default=True, help="save generated accuracy plots after training"
    )

    return parser


def init_optim(
    model,
    args,
    ):
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), args.lr, args.momentum, args.weight_decay)
        scheduler_optim = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.9)  # the scheduler divides the lr by 10 every 10 epochs
        return (optimizer, scheduler_optim)
    elif args.optimizer == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), args.lr, args.alpha, 1e-08, args.momentum, args.weight_decay)
        return optimizer
    else:
        # print(args.betas)
        optimizer = optim.Adam(model.parameters(), args.lr, args.betas, 1e-08, args.weight_decay)
        return optimizer
