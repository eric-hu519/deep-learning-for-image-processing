#!/usr/bin/env python
import wandb
import os
import random

sweep_configuration = {
    'method': 'grid',
    'name': 'sweep-test-1',
    'metric': {
        'goal': 'maximize',
        'name': 'val_accuracy'
    },
    'parameters': {
        'batch_size': {'values': [16, 32]},
    }
}


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for key in os.environ.keys():
        if key.startswith("WANDB_") and key not in exclude:
            del os.environ[key]


def train(num, sweep_id, sweep_run_name, config):
    run_name = f'{sweep_run_name}-{num}'
    run = wandb.init(
        group=sweep_id,
        job_type=sweep_run_name,
        name=run_name,
        config=config,
        reinit=True
    )
    val_accuracy = random.random()
    run.log(dict(val_accuracy=val_accuracy))
    run.finish()
    return val_accuracy


def cross_validate():
    num_folds = 3

    sweep_run = wandb.init()
    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_url = sweep_run.get_sweep_url()
    project_url = sweep_run.get_project_url()
    sweep_group_url = f'{project_url}/groups/{sweep_id}'
    sweep_run.notes = sweep_group_url
    sweep_run.save()
    sweep_run_name = sweep_run.name or sweep_run.id or "unknown_2"
    sweep_run_id = sweep_run.id
    sweep_run.finish()
    wandb.sdk.wandb_setup._setup(_reset=True)

    metrics = []
    for num in range(num_folds):
        reset_wandb_env()
        result = train(
            sweep_id=sweep_id,
            num=num,
            sweep_run_name=sweep_run_name,
            config=dict(sweep_run.config),
        )
        metrics.append(result)

    # resume the sweep run
    sweep_run = wandb.init(id=sweep_run_id, resume="must")
    # log metric to sweep run
    sweep_run.log(dict(val_accuracy=sum(metrics) / len(metrics)))
    sweep_run.finish()

    print("*" * 40)
    print("Sweep URL:       ", sweep_url)
    print("Sweep Group URL: ", sweep_group_url)
    print("*" * 40)


def main():
    wandb.login()
    sweep_id = wandb.sweep(sweep_configuration, project='sweep-test')
    wandb.agent(sweep_id, function=cross_validate)

    wandb.finish()


if __name__ == "__main__":
    main()
