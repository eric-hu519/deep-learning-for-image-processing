import os
import wandb
import re
import glob
from pathlib import Path

def wandb_init(args):
    wandb.init(project="spinopelvic", config=args)
    wandb.run.name = str(args.output_dir).split('/')[-1]
    #wandb.run.save()
    wandb.watch_called = False

    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config          # Initialize config
    config.learning_rate = args.lr
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.weight_decay = args.weight_decay
    config.lr_steps = args.lr_steps
    config.lr_gamma = args.lr_gamma
    config.num_joints = args.num_joints
    config.fixed_size = args.fixed_size
    config.with_FFCA = args.with_FFCA

def wandb_log(epoch, train_loss, cocoinfo,best_results):
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": cocoinfo[14],
        "sc_abs_error": cocoinfo[10],
        "s1_abs_error": cocoinfo[11],
        "fh1_abs_error": cocoinfo[12],
        "fh2_abs_error": cocoinfo[13],
        "sc_best": best_results[0],
        "s1_best": best_results[1],
        "fh1_best": best_results[2],
        "fh2_best": best_results[3],

        "Avg Precision @[ IoU=0.50:0.95 | area=   all | maxDets=20 ]": cocoinfo[0],
        "Avg Precision @[ IoU=0.50      | area=   all | maxDets=20 ]": cocoinfo[1],
        "Avg Precision @[ IoU=0.75      | area=   all | maxDets=20 ]": cocoinfo[2],
        "Avg Precision @[ IoU=0.50:0.95 | area= large | maxDets=20 ]": cocoinfo[4],
        "Avg Recall @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]": cocoinfo[5],
        "Avg Recall @[ IoU=0.50 | area= all | maxDets= 20 ]": cocoinfo[6],
        "Avg Recall @[ IoU=0.75 | area= all | maxDets= 20 ]": cocoinfo[7],
        "Avg Recall @[ IoU=0.50:0.95 | area= large | maxDets= 20 ]": cocoinfo[9],
    })

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def sweep_override(args):
    #limit_batch to avoid cuda overdrive
    if args.fixed_size < 512:
        args.batch_size = 32 
    elif args.fixed_size == 512 :
        args.batch_size = 28
    elif args.fixed_size > 512 :
        args.batch_size = 16
    #adjust lr scheme according to args
    if args.lr_steps == 1:
        stage1 = round((args.epochs-1) * 0.25)
        stage2 = round((args.epochs-1) * 0.5)
        args.lr_steps = [stage1,stage2]
    elif args.lr_steps == 2:
        stage1 = round((args.epochs-1) * 0.5)
        stage2 = round((args.epochs-1) * 0.75)
        args.lr_steps = [stage1,stage2]
    elif args.lr_steps == 3:
        stage1 = round((args.epochs-1) * 0.25)
        stage2 = round((args.epochs-1) * 0.75)
        args.lr_steps = [stage1,stage2]
    elif args.lr_steps == 4:
        stage1 = round((args.epochs-1) * 0.5)
        stage2 = args.epoch-1
        args.lr_steps = [stage1,stage2]
    elif args.lr_steps == 5:
        stage1 = round((args.epochs-1) * 0.75)
        stage2 = args.epoch-1
        args.lr_steps = [stage1,stage2]
    elif args.lr_steps == 6:
        stage1 = round((args.epochs-1) * 0.25)
        stage2 = args.epoch-1
        args.lr_steps = [stage1,stage2]
    return args
    
def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for key in os.environ.keys():
        if key.startswith("WANDB_") and key not in exclude:
            del os.environ[key]
