import os
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import torchvision.models as models
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR


import utils
import prune_wrapper


def log_metrics(metric_dict,epoch,title):
    # Log all metrics in dictionary to tensorboardX
    for key in metric_dict.keys():
        label_text = "{}/{}".format(title,key)
        logger.add_scalar(label_text,metric_dict[key],epoch)


def train():
    # 1. Set network to be in training mode
    net.train()
    # 2. Setup loggers, and progress bar
    correct = 0
    total_loss = 0
    total_num = 1
    pbar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc='Train epoch {}'.format(epoch))
    # 3. Iterate through
    for batch_idx, (data, target) in pbar:
        # 3a. Set to correct device
        data, target = data.to(device), target.to(device)
        # 3b. Zero out previous gradients
        optimizer.zero_grad()
        # 3c. Forward propagate
        outputs = net(data)
        # 3d. Calculate loss
        loss = loss_fn(outputs, target)
        loss_val = loss.item()
        # 3e. Backpropagate loss
        loss.backward()
        # 3f. Take step with optimizer
        optimizer.step()
        # 3g. Update loggers
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == target).sum().item()
        total_loss += loss_val
        total_num += data.shape[0]
        # 3h. Update progress bar
        pbar.set_postfix(avg_acc=float(correct)/total_num)
    # 4. Calculate average loss and accuracy and return
    acc = float(correct)/total_num
    avg_loss = total_loss/(batch_idx+ 1) # Average per batch
    return {"train_avg_loss": avg_loss, "train_acc":acc}

def validate():
    # 1. Set network to be in evaluation mode
    net.eval()
    # 2. Setup loggers, and progress bar
    correct = 0
    total_loss = 0
    total_num = 0
    pbar = tqdm(enumerate(validation_data_loader), total=len(validation_data_loader), desc='Validation epoch {}'.format(epoch))
    # 3. Iterate through, with gradients off
    with torch.no_grad():
        for batch_idx, (data, target) in pbar:
            # 3a. Set to correct device
            data, target = data.to(device), target.to(device)
            # 3b. Forward propagate
            outputs = net(data)
            # 3c. Calculate loss
            loss = loss_fn(outputs, target)
            loss_val = loss.item()
            # 3d. Update loggers
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == target).sum().item()
            total_loss += loss_val
            total_num += data.shape[0]
            # 3f. Update progress bar
            pbar.set_postfix(avg_acc=float(correct)/total_num)
    # 4. Calculate average loss and accuracy and return
    acc = float(correct)/total_num
    avg_loss = total_loss/(batch_idx+ 1) # Average per batch
    return {"validate_avg_loss": avg_loss, "validate_acc":acc}

def test():
    # 1. Set network to be in evaluation mode
    net.eval()
    # 2. Setup loggers, and progress bar
    correct = 0
    total_loss = 0
    total_num = 0
    pbar = tqdm(enumerate(test_data_loader), total=len(test_data_loader), desc='Test epoch {}'.format(epoch))
    # 3. Iterate through, with gradients off
    with torch.no_grad():
        for batch_idx, (data, target) in pbar:
            # 3a. Set to correct device
            data, target = data.to(device), target.to(device)
            # 3b. Forward propagate
            outputs = net(data)
            # 3c. Calculate loss
            loss = loss_fn(outputs, target)
            loss_val = loss.item()
            # 3d. Update loggers
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == target).sum().item()
            total_loss += loss_val
            total_num += data.shape[0]
            # 3f. Update progress bar
            pbar.set_postfix(avg_acc=float(correct)/total_num)
    # 4. Calculate average loss and accuracy and return
    acc = float(correct)/total_num
    avg_loss = total_loss/(batch_idx+ 1) # Average per batch
    return {"test_avg_loss": avg_loss, "test_acc":acc}

RANDOM_SEED = 1994
if __name__ == "__main__":
    # 1. Load arguments from command line
    parser = argparse.ArgumentParser(description='Process parameters for network pruning')
    parser.add_argument('--train_epochs', '-te', type=int, default=45, help='number of epochs to train each pruning round')
    parser.add_argument('--late_reset_epoch', '-lr', type=int, default=0, help='if using late reseting the epoch to save after, default of 0 means no late reseting')
    parser.add_argument('--num_pruning_rounds', '-npr', type=int, default=25, help='number of pruning rounds')
    parser.add_argument('--prune_rate', '-pr', type=float, default=.2, help='percentage of weights to prune each iteration')
    parser.add_argument('--random_initialization', '-ri', action='store_true',help="randomly reinitalize instead of using same initialization")
    parser.add_argument('--random_prune', '-rp', action='store_true',help="randomly prune weights instead of using weight magnitude")
    parser.add_argument('--dataset', '-ds', type=str, default="svhn", help='which dataset to use',choices=['cifar100','svhn'])
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    if args.verbose: print(args)

    # 2. Seed random processes
    np.random.seed(RANDOM_SEED)

    # 3. Set Torch device and if GPU set to optimize graph for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.verbose: print("Using device: {}".format(device))
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # 4. Setup logging directory and Tensorboard writer
    dt_string = datetime.now().strftime("%d%m%Y%H%M%S")
    dir_name = "runs/pruning_experiment_DS:{}_PR:{}_LR:{}_RI:{}_RP:{}_{}".format(args.dataset,args.prune_rate,args.late_reset_epoch,args.random_initialization,args.random_prune,dt_string)
    logger = SummaryWriter(log_dir=dir_name)

    # 5. Setup data loaders
    train_data_loader, validation_data_loader, test_data_loader, num_classes = utils.build_data_loaders(args.dataset,batch_size=512)

    # 6. Setup network
    net = models.resnet50(pretrained=False, num_classes=num_classes)
    net.to(device)

    # 7. Setup epochs, optimize, steps
    num_epochs = args.train_epochs
    epoch_steps = [int(.5*num_epochs),int(.75*num_epochs)]
    loss_fn = nn.CrossEntropyLoss()


    # 8. Now setup prunging wrapper
    name_filter = lambda x : "weight" in x and "fc" not in x
    net = prune_wrapper.PruningWrapper(net,args.prune_rate,device,dir_name,random_initialization=args.random_initialization,random_prune=args.random_prune,filter_function=name_filter)

    # 9. Begin pruning loop
    logging_dict = {}
    for round_num in range(args.num_pruning_rounds):
        # 9a. Setup learning rate scheduler
        optimizer = optim.SGD(net.parameters(), lr=.1,momentum=.9,weight_decay=0.0001)
        lr_schedule = MultiStepLR(optimizer, milestones=epoch_steps, gamma=0.1)
        # 9b. Now train for set number of epochs
        best_validation_acc = 0
        for epoch in range(num_epochs):
            if round_num == 0 and epoch == args.late_reset_epoch:
                net.set_initial_weights()
            # Train and Test one epoch get loss and acc
            train_dict = train()
            validate_dict = validate()
            # If has best accuracy update
            if validate_dict["validate_acc"] > best_validation_acc:
                best_validate_acc = validate_dict["validate_acc"]
                best_validate_acc_total_metrics = {**train_dict, **validate_dict}
                torch.save(net.state_dict(),os.path.join(dir_name,"current_best_model.pth"))
            # 9f. Step learning rate scheduler
            lr_schedule.step()
            total_epoch_metrics = {**train_dict, **validate_dict}
            log_metrics(total_epoch_metrics,epoch,"round_{}".format(round_num))
        # 9d. Use best validation network to caluclate test accuracy
        net.load_state_dict(torch.load(os.path.join(dir_name,"current_best_model.pth")))
        test_dict = test()

        # 9c. Now log with percent removed
        tu,tp,percent_pruned = net.calculate_percent_pruned()
        added_metrics = {"total_unpruned":tu,"total_prunable":tp,"percent_pruned":percent_pruned}
        total_metrics = {**added_metrics, **test_dict}
        log_metrics(total_metrics,round_num,"round_num_vs")
        logging_dict["round_{}".format(round_num)] = total_metrics
        if args.verbose: print("{} percent pruned, {} Acc".format(percent_pruned,total_metrics["test_acc"]))
        # 9d. Now prune weights, and reset to init
        net.prune_weights()
        net.reinitialize()

    # 10. Close logger, and save logging dictionary
    logger.close()
    output = open(os.path.join(dir_name,"full_logs.pkl"), 'wb')
    pickle.dump(logging_dict, output)
    output.close()










 #
