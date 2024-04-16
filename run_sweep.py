import wandb

sweep_configuration = {
    "program": "run_agent.py",
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "avg_score"},
    "parameters": {
        "acquisition_function": {"values": ["general_ucb", "general_ei"]},
        "v_confidence_level": {"min": 0.9, "max": 0.9999},
        "f_confidence_level": {"min": 0.2, "max": 0.99},
        "f_confidence_level_decay": {"value": 0},
        "reward_coef": {"min": 0.1, "max": 10.0},
        "safety_coef": {"min": 0.0, "max": 100.0},
        "safety_constraint_coef": {"min": 0.0, "max": 100.0},
        "f_kernel": {"value": "matern"},
        "v_kernel": {"value": "rbf"},
        "v_length_scale": {"value": 10},
        "default_to_safety": {"values": [True, False]},
        "pertubation_step": {"min": 0.0001, "max": 0.1},
    },
}

wandb.sweep(sweep_configuration, project="pai-task3", entity="eugleo")
