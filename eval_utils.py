# ------------------------------------------------------------------------------
#    Functions to evaluate, visualize and save model results
# ------------------------------------------------------------------------------
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from options import create_parser, init_optim

def count_parameters(model):
    table = PrettyTable(["Modules", "Size", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        size = list(parameter.shape)
        table.add_row([name, size, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}\n")


def evaluate(model, test_dataloaders, accs_dict, avg_acc_list, avg_loss_list, loss_fn, device, task_nr, verbose=True):
    # Evaluating tasks seen so far
    acc_sum, loss_sum = 0.0, 0.0

    for t_s in range(task_nr + 1):
        # Test model
        if model.scenario == "class":
            acc, loss = model.test_epoch(test_dataloaders[t_s], loss_fn, device, task_nr=task_nr, verbose=verbose)
            # acc, loss = model.test_epoch(test_dataloaders[t_s], loss_fn, device, task_nr=t_s, verbose=verbose)
        else:
            acc, loss = model.test_epoch(test_dataloaders[t_s], loss_fn, device, task_nr=t_s, verbose=verbose)

        accs_dict["task_" + str(t_s + 1)].append(acc)
        acc_sum += acc
        loss_sum += loss

    # Update average accuracy
    avg_acc_list.append(acc_sum / (task_nr + 1))
    # Update average test loss
    avg_loss_list.append(loss_sum / (task_nr + 1))

    return accs_dict, avg_acc_list, avg_loss_list


def save_model(task_id, epoch, pruner_model, base_model, metrics, mode):
    parser = create_parser()
    args, unknown = parser.parse_known_args()

    path = f"./checkpoints/cl_r2plus1d_model_taskid_{task_id}_{mode}.pth.tar"

    pruner_model_copy = deepcopy(pruner_model)
    base_model_copy = deepcopy(base_model)

    if mode=='retrain':
        pruner_model_copy.make_finetuning_mask()

    torch.save({
        'last_task_id': task_id,
        'learning_status':mode,
        'best_epoch':epoch,
        'pruner_state_dict': pruner_model_copy.model.state_dict(),
        'pruner_optimizer': pruner_model_copy.model.optimizer,
        'previous_masks':pruner_model_copy.previous_masks,
        'current_masks':pruner_model_copy.current_masks,
        'base_state_dict': base_model_copy.state_dict(),
        'base_optimizer': base_model_copy.optimizer,
        'perf_metrics': metrics,
    }, path)

def task_learning_checkpointing(acc, best_acc, best_epoch, es_thresh, epoch, pruner_model, base_model, metrics, task_id, mode):
    # final_epoch = epoch + es_thresh
    # emergency checkpoint saving
    emerg_path = f"./checkpoints/last_status.pth.tar"
    torch.save({
        'last_task_id': task_id,
        'learning_status':mode,
        'last_epoch':epoch,
        'perf_metrics': metrics,
    }, emerg_path)

    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
        save_model(task_id, best_epoch, pruner_model, base_model, metrics, mode)
        return 'no'
    elif epoch - best_epoch > es_thresh:
        print("Early stopped training at epoch %d" % epoch)
        return 'yes'  # terminate the training loop
    else:
        return 'no'


def plot_list(items_list, y_desc="", title="", save=False, filename=""):
    # Plot elements from list
    plt.figure()
    for item in items_list:
        if len(item) > 0:
            plt.plot(item)
    plt.xlabel("Epoch"), plt.ylabel(y_desc)
    plt.grid()
    plt.title(title)
    plt.legend(["CL model", "base_model"])

    # Save file
    if save:
        plt.savefig("{}.png".format(filename), dpi=300)
    # plt.show()


def plot_dict(acc_dict, epochs, tasks, y_desc="", title="", save=False, filename=""):
    # Plot elements from dictionary
    plt.figure()
    for i, (task_key, acc_list) in enumerate(acc_dict.items()):
        x = np.arange(i * epochs, tasks * epochs)
        plt.plot(x, acc_list, label=task_key)
    plt.xlabel("Epoch"), plt.ylabel(y_desc)
    plt.title(title)
    plt.grid(), plt.legend()

    # Save file
    if save:
        plt.savefig("{}.png".format(filename), dpi=300)
    # plt.show()
