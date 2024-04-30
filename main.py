import os
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from collections import OrderedDict
import models
import copy
from data.poison_tool_cifar import get_test_loader, get_train_loader
from torch.utils.data import random_split, DataLoader, Dataset

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# seed = 98
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.manual_seed(seed)
# np.random.seed(seed)


def train_step_unlearning(args, model, criterion, optimizer, data_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        (-loss).backward()
        optimizer.step()

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc

def train_step_recovering(args, unlearned_model, criterion, mask_opt, data_loader):
    unlearned_model.train()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        nb_samples += images.size(0)

        mask_opt.zero_grad()
        output = unlearned_model(images)
        loss = criterion(output, labels)
        loss = args.alpha * loss

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()
        mask_opt.step()
        clip_mask(unlearned_model)

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc


def load_state_dict(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)


def clip_mask(unlearned_model, lower=0.0, upper=1.0):
    params = [param for name, param in unlearned_model.named_parameters() if 'neuron_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def save_mask_scores(state_dict, file_name):
    mask_values = []
    count = 0
    for name, param in state_dict.items():
        if 'neuron_mask' in name:
            for idx in range(param.size(0)):
                neuron_name = '.'.join(name.split('.')[:-1])
                mask_values.append('{} \t {} \t {} \t {:.4f} \n'.format(count, neuron_name, idx, param[idx].item()))
                count += 1
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask Score \n')
        f.writelines(mask_values)

def read_data(file_name):
    tempt = pd.read_csv(file_name, sep='\s+', skiprows=1, header=None)
    layer = tempt.iloc[:, 1]
    idx = tempt.iloc[:, 2]
    value = tempt.iloc[:, 3]
    mask_values = list(zip(layer, idx, value))
    return mask_values

def pruning(net, neuron):
    state_dict = net.state_dict()
    weight_name = '{}.{}'.format(neuron[0], 'weight')
    state_dict[weight_name][int(neuron[1])] = 0.0
    net.load_state_dict(state_dict)

def test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc

def evaluate_by_number(model, logger, mask_values, pruning_max, pruning_step, criterion, clean_loader, poison_loader):
    results = []
    nb_max = int(np.ceil(pruning_max))
    nb_step = int(np.ceil(pruning_step))
    for start in range(0, nb_max + 1, nb_step):
        i = start
        for i in range(start, start + nb_step):
            pruning(model, mask_values[i])
        layer_name, neuron_idx, value = mask_values[i][0], mask_values[i][1], mask_values[i][2]
        cl_loss, cl_acc = test(model=model, criterion=criterion, data_loader=clean_loader)
        po_loss, po_acc = test(model=model, criterion=criterion, data_loader=poison_loader)
        logger.info('{} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            i+1, layer_name, neuron_idx, value, po_loss, po_acc, cl_loss, cl_acc))
        results.append('{} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            i+1, layer_name, neuron_idx, value, po_loss, po_acc, cl_loss, cl_acc))
    return results


def evaluate_by_threshold(model, logger, mask_values, pruning_max, pruning_step, criterion, clean_loader, poison_loader):
    results = []
    thresholds = np.arange(0, pruning_max + pruning_step, pruning_step)
    start = 0
    for threshold in thresholds:
        idx = start
        for idx in range(start, len(mask_values)):
            if float(mask_values[idx][2]) <= threshold:
                pruning(model, mask_values[idx])
                start += 1
            else:
                break
        layer_name, neuron_idx, value = mask_values[idx][0], mask_values[idx][1], mask_values[idx][2]
        cl_loss, cl_acc = test(model=model, criterion=criterion, data_loader=clean_loader)
        po_loss, po_acc = test(model=model, criterion=criterion, data_loader=poison_loader)
        logger.info('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            start, layer_name, neuron_idx, threshold, po_loss, po_acc, cl_loss, cl_acc))
        results.append('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}\n'.format(
            start, layer_name, neuron_idx, threshold, po_loss, po_acc, cl_loss, cl_acc))
    return results

def save_checkpoint(state, file_path):
    # filepath = os.path.join(args.output_dir, args.arch + '-unlearning_epochs{}.tar'.format(epoch))
    torch.save(state, file_path)

def check_zero_weights(model):
    total_weights = 0
    zero_weights = 0
    for param in model.parameters():
        total_weights += param.numel()
        zero_weights += (param == 0).sum().item()
    print(f"Total weights: {total_weights}, Zero weights: {zero_weights} ({100 * zero_weights / total_weights:.2f}%)")

def prune_based_on_RNP_unlearn_signal(model1, model2, input, prune_ratio=0.3, prune_all_layers=False):
    model1 = copy.deepcopy(model1)
    model2 = copy.deepcopy(model2)

    if not prune_ratio > 0.0:
        return model1, model2

    # Dictionary to store activations keyed by layer names
    activations_store = {}
    activation_diffs = {}

    def get_layer_path(module, prefix=''):
        """ Recursively get the path for a layer within the model's hierarchy. """
        layer_path = {module: prefix}
        for name, child in module.named_children():
            child_path = get_layer_path(child, prefix=f"{prefix}/{name}" if prefix else name)
            layer_path.update(child_path)
        return layer_path

    # Get layer paths
    layer_paths1 = get_layer_path(model1)
    layer_paths2 = get_layer_path(model2)

    def forward_hook1(module, inp, out):
        layer_name = layer_paths1[module]
        activations_store[layer_name] = out.detach()
        print(f"Activations1 recorded for layer: {layer_name}")

    def forward_hook2(module, inp, out):
        layer_name = layer_paths2[module]
        if layer_name in activations_store:
            activation_diffs[layer_name] = torch.abs(activations_store[layer_name] - out)
            print(f"Difference calculated for layer: {layer_name}")
        else:
            print(f"No previous activations found for layer: {layer_name}")

    # Register hooks for both models based on the boolean input
    hooks1 = []
    hooks2 = []
    if prune_all_layers:
        for module in layer_paths1:
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks1.append(module.register_forward_hook(forward_hook1))
        for module in layer_paths2:
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks2.append(module.register_forward_hook(forward_hook2))
    else:
        # Find the second-to-last convolutional or linear layer
        eligible_layers1 = [m for m in model1.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        eligible_layers2 = [m for m in model2.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        if len(eligible_layers1) > 1 and len(eligible_layers2) > 1:
            module1 = eligible_layers1[-2]
            module2 = eligible_layers2[-2]
            hooks1.append(module1.register_forward_hook(forward_hook1))
            hooks2.append(module2.register_forward_hook(forward_hook2))

    model1.eval()
    model2.eval()
    with torch.no_grad():
        _ = model1(input)
        _ = model2(input)

    # Remove hooks
    for hook in hooks1:
        hook.remove()
    for hook in hooks2:
        hook.remove()

    total_filters = 0
    pruned_filters = 0
    # Apply pruning based on calculated differences
    for layer_name, diffs in activation_diffs.items():
        if diffs.dim() == 4:  # Conv2d layers
            importance_scores = diffs.mean(dim=[0, 2, 3])
        elif diffs.dim() == 2:  # Linear layers
            importance_scores = diffs.mean(dim=0)
        else:
            continue

        num_filters = importance_scores.size(0)
        total_filters += num_filters
        threshold = torch.quantile(importance_scores, prune_ratio)
        prune_mask = importance_scores < threshold
        pruned_filters += prune_mask.sum().item()

        # Zero out weights based on the pruning mask
        module1 = next(m for m in model1.modules() if layer_paths1[m] == layer_name)
        module2 = next(m for m in model2.modules() if layer_paths2[m] == layer_name)
        module1.weight.data[prune_mask, ...] = 0
        module2.weight.data[prune_mask, ...] = 0
        if module1.bias is not None:
            module1.bias.data[prune_mask] = 0
        if module2.bias is not None:
            module2.bias.data[prune_mask] = 0

        print(f"Pruning applied to layers: {layer_name} | Pruned filters in this layer: {prune_mask.sum().item()}")

    print(f"Total filters: {total_filters}, Pruned filters: {pruned_filters}")
    return model1, model2

def main(args):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.log_root, 'output.log')),
            logging.StreamHandler()
        ])
    logger.info(args)

    logger.info('----------- Data Initialization --------------')
    defense_data, defense_data_loader = get_train_loader(args)
    clean_test_loader = get_test_loader(args)

    logger.info('----------- Backdoor Model Initialization --------------')
    state_dict = torch.load(args.backdoor_model_path, map_location=device)["model"]
    if args.arch == "resnet18":
        from torchvision.models.resnet import resnet18
        net = resnet18(num_classes=10, norm_layer=None)
    else:
        net = getattr(models, args.arch)(num_classes=10, norm_layer=None)
    load_state_dict(net, orig_state_dict=state_dict)
    net = net.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.unlearning_lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)

    unlearned_net = copy.deepcopy(net)

    logger.info('----------- Model Unlearning --------------')
    logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    for epoch in range(0, args.unlearning_epochs + 1):
        start = time.time()
        lr = optimizer.param_groups[0]['lr']
        train_loss, train_acc = train_step_unlearning(args=args, model=unlearned_net, criterion=criterion, optimizer=optimizer,
                                      data_loader=defense_data_loader)
        cl_test_loss, cl_test_acc = test(model=unlearned_net, criterion=criterion, data_loader=clean_test_loader)
        scheduler.step()
        end = time.time()
        logger.info(
            '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, lr, end - start, train_loss, train_acc,
            cl_test_loss, cl_test_acc)

        if train_acc <= args.clean_threshold:

            print(f"epoch: {epoch}")
            print(f"clean_acc: {cl_test_acc}")
            break

    defense_loader = DataLoader(defense_data, batch_size=len(defense_data), shuffle=True)

    for inputs, targets in defense_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        print(f"inputs.size: {inputs.size}")
        break  # Assuming we use only one batch for the example

    for prune_ratio in args.prune_ratio_list:
        print()
        pruned_model, _ = prune_based_on_RNP_unlearn_signal(net, unlearned_net, inputs,
                                                                  prune_ratio=prune_ratio,
                                                                  prune_all_layers=args.prune_all_layers)

        # model = resnet18(pretrained=True)
        print("model")
        check_zero_weights(net)

        print("pruned_model")
        check_zero_weights(pruned_model)

        cl_test_loss, cl_test_acc = test(model=pruned_model, criterion=criterion, data_loader=clean_test_loader)

        print(f"prune_ratio: {prune_ratio}")
        print(f"clean_acc: {cl_test_acc}")

        file_path = os.path.join(args.output_weight, f'RNP_unlearn_signal_pruned_model{args.unlearn_file_suffix}.pt')

        save_checkpoint({
            'model': pruned_model.state_dict(),
            'clean_acc': cl_test_acc,
        }, file_path)

if __name__ == '__main__':
    # Prepare arguments
    parser = argparse.ArgumentParser()

    # various path
    parser.add_argument('--cuda', type=int, default=1, help='cuda available')
    parser.add_argument('--save-every', type=int, default=5, help='save checkpoints every few epochs')
    parser.add_argument('--log_root', type=str, default='logs/', help='logs are saved here')
    parser.add_argument('--output_weight', type=str, default='weights/')
    parser.add_argument('--unlearn_file_suffix', type=str, default='')
    parser.add_argument('--backdoor_model_path', type=str,
                        default='weights/ResNet18-ResNet-BadNets-target0-portion0.1-epoch80.tar',
                        help='path of backdoored model')
    parser.add_argument('--unlearned_model_path', type=str,
                        default=None, help='path of unlearned backdoored model')
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2',
                                 'vgg19_bn'])
    parser.add_argument('--schedule', type=int, nargs='+', default=[10, 20],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of image dataset')
    parser.add_argument('--input_height', type=int, default=32, help='The size of batch')
    parser.add_argument('--input_width', type=int, default=32, help='The size of batch')
    parser.add_argument('--prune_ratio_list', type=float, nargs="+")
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--num_class', type=int, default=10, help='number of classes')
    parser.add_argument('--ratio', type=float, default=0.01, help='ratio of defense data')

    # backdoor attacks
    parser.add_argument('--target_label', type=int, default=0, help='class of target label')
    parser.add_argument('--trigger_type', type=str, default='gridTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')

    # RNP
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--clean_threshold', type=float, default=0.20, help='threshold of unlearning accuracy')
    parser.add_argument('--unlearning_lr', type=float, default=0.01, help='the learning rate for neuron unlearning')
    parser.add_argument('--recovering_lr', type=float, default=0.2, help='the learning rate for mask optimization')
    parser.add_argument('--unlearning_epochs', type=int, default=20, help='the number of epochs for unlearning')
    parser.add_argument('--recovering_epochs', type=int, default=20, help='the number of epochs for recovering')
    parser.add_argument('--mask_file', type=str, default=None, help='The text file containing the mask values')
    parser.add_argument('--pruning-by', type=str, default='threshold', choices=['number', 'threshold'])
    parser.add_argument('--pruning-max', type=float, default=0.90, help='the maximum number/threshold for pruning')
    parser.add_argument('--pruning-step', type=float, default=0.05, help='the step size for evaluating the pruning')

    args = parser.parse_args()
    args_dict = vars(args)
    print(args_dict)
    os.makedirs(args.log_root, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    main(args)
