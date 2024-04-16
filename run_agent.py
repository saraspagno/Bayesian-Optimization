import json
import re
import subprocess

import wandb


def main():
    wandb.init(project="pai-task3", entity="eugleo")

    with open("config.json", "w") as f:
        config = {
            "acquisition_function": wandb.config.acquisition_function,
            "v_confidence_level": wandb.config.v_confidence_level,
            "f_confidence_level": wandb.config.f_confidence_level,
            "f_confidence_level_decay": wandb.config.f_confidence_level_decay,
            "reward_coef": wandb.config.reward_coef,
            "safety_coef": wandb.config.safety_coef,
            "safety_constraint_coef": wandb.config.safety_constraint_coef,
            "f_kernel": wandb.config.f_kernel,
            "v_kernel": wandb.config.v_kernel,
            "v_length_scale": wandb.config.v_length_scale,
            "default_to_safety": wandb.config.default_to_safety,
            "pertubation_step": wandb.config.pertubation_step,
        }
        json.dump(config, f)

    process = subprocess.Popen(["bash", "runner.sh"], text=True, stdout=subprocess.PIPE)

    lines = []
    step = 0
    while True:
        output = process.stdout.readline()
        if output == "" and process.poll() is not None:
            break
        if output:
            line = output.strip()
            lines.append(line)
            if line.startswith("Regret of problem"):
                regret = float(line.split(":")[-1].strip())
                wandb.log({"regret": regret}, step=step)
            elif line.startswith("Unsafe evaluations"):
                unsafe_evals = int(line.split(":")[-1].strip())
                wandb.log({"unsafe_evaluations": unsafe_evals}, step=step)
            elif line.startswith("Score"):
                score = float(line.split(":")[-1].strip())
                wandb.log({"score": score}, step=step)
                step += 1

    matches = re.findall(r"([\d\.]+)\.", " ".join(lines[-3:]))
    if not matches:
        print("Could not find final score information: ", lines[-3:])
    else:
        score = float(matches[0])
        wandb.summary["avg_score"] = score

        artifact = wandb.Artifact("results_check", type="output")
        artifact.add_file("results_check.byte")
        wandb.log_artifact(artifact)


if __name__ == "__main__":
    main()
