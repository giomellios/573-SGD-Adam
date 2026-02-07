import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT_DIRS = ["SGD_RUNS", "ADAM_RUNS"]

LAYER_ORDER = ["grad/conv1", "grad/layer1", "grad/layer2", "grad/layer3"]
LAYER_LABELS = ["Conv1", "Layer 1", "Layer 2", "Layer 3"]

os.makedirs("plots", exist_ok=True)
os.makedirs("basins", exist_ok=True)

sns.set(style="whitegrid")


def parse_logs_for_plots(root_dirs):

    grad_data = []
    gen_data = []
    final_perf_data = []

    print("Scanning logs...")

    for root_dir in root_dirs:
        if not os.path.exists(root_dir):
            continue

        for dataset in os.listdir(root_dir):

            ds_path = os.path.join(root_dir, dataset)
            if not os.path.isdir(ds_path):
                continue

            for run_folder in os.listdir(ds_path):

                run_path = os.path.join(ds_path, run_folder)
                if not os.path.isdir(run_path):
                    continue

                event_files = glob.glob(
                    os.path.join(run_path, "events.out.tfevents*")
                )
                if not event_files:
                    continue

        
        
        

                try:
                    parts = run_folder.split("_")

                    optimizer = (
                        "Adam" if "ADAM" in root_dir.upper() else "SGD"
                    )

                    arch = parts[0]

                    bs = int(
                        next(p for p in parts if p.startswith("BS"))
                        .replace("BS", "")
                    )

                    lr = float(
                        next(p for p in parts if p.startswith("LR"))
                        .replace("LR", "")
                    )

                except Exception as e:
                    print(f"Skipping malformed run name: {run_folder}")
                    continue

                ea = EventAccumulator(event_files[0])
                ea.Reload()
                tags = ea.Tags()["scalars"]

                if "grad/conv1" in tags:

                    available_steps = [
                        s.step for s in ea.Scalars("grad/conv1")
                    ]

                    if available_steps:

                        target_step = available_steps[
                            min(len(available_steps) - 1, 5)
                        ]

                        for i, layer_tag in enumerate(LAYER_ORDER):

                            if layer_tag in tags:

                                events = ea.Scalars(layer_tag)

                                val = next(
                                    (
                                        e.value
                                        for e in events
                                        if e.step == target_step
                                    ),
                                    events[0].value,
                                )
                                print(arch)
                                grad_data.append(
                                    {
                                        "Optimizer": optimizer,
                                        "Architecture": arch,
                                        "Dataset": dataset,
                                        "Layer": LAYER_LABELS[i],
                                        "LayerDepth": i,
                                        "Norm": val,
                                        "BatchSize": bs,
                                        "LR": lr,
                                    }
                                )

                if "epoch/gen_gap" in tags:

                    events = ea.Scalars("epoch/gen_gap")

                    for e in events:
                        gen_data.append(
                            {
                                "Optimizer": optimizer,
                                "Architecture": arch,
                                "Dataset": dataset,
                                "BatchSize": bs,
                                "LR": lr,
                                "Epoch": e.step,
                                "GenGap": e.value,
                            }
                        )

                test_tag = "epoch/test_acc"
                train_tag = "epoch/train_acc"

                if test_tag in tags and train_tag in tags:

                    test_events = ea.Scalars(test_tag)
                    train_events = ea.Scalars(train_tag)

                    final_test = test_events[-1].value
                    final_train = train_events[-1].value

                    diverged = (
                        np.isnan(final_test)
                        or final_test < 0.15
                    )

                    acc_thresh = (
                        0.95
                        if dataset.lower().startswith("mnist")
                        else 0.80
                    )

                    epochs_to_thresh = None
                    for e in test_events:
                        if e.value >= acc_thresh:
                            epochs_to_thresh = e.step
                            break

                    final_perf_data.append(
                        {
                            "Optimizer": optimizer,
                            "Architecture": arch,
                            "Dataset": dataset,
                            "BatchSize": bs,
                            "LR": lr,
                            "FinalTestAcc": final_test,
                            "FinalTrainAcc": final_train,
                            "GenGapFinal": final_test
                            - final_train,
                            "Diverged": diverged,
                            "EpochsToThresh": epochs_to_thresh,
                        }
                    )

    return (
        pd.DataFrame(grad_data),
        pd.DataFrame(gen_data),
        pd.DataFrame(final_perf_data),
    )


def plot_signal_propagation(df):

    subset = df[df["Architecture"] == "ResNet"]

    if subset.empty:
        print("No PlainNet gradient data.")
        return

    for ds in subset["Dataset"].unique():

        ds_sub = subset[subset["Dataset"] == ds]

        plt.figure(figsize=(9, 6))

        sns.lineplot(
            data=ds_sub,
            x="Layer",
            y="Norm",
            hue="Optimizer",
            style="Optimizer",
            markers=True,
            dashes=False,
        )

        plt.yscale("log")
        plt.title(
            f"Early Gradient Flow — ResNet — {ds}"
        )
        plt.ylabel("Gradient Norm (log)")
        plt.xlabel("Layer Depth")
        plt.tight_layout()

        plt.savefig(f"plots/GradientFlow_{ds}.png")
        plt.close()


def plot_gen_gap(df):

    if df.empty:
        return

    target_bs = df["BatchSize"].max()
    subset = df[df["BatchSize"] == target_bs]

    for ds in subset["Dataset"].unique():

        ds_sub = subset[
            (subset["Dataset"] == ds)
            & (subset["Architecture"] == "ResNet")
        ]

        if ds_sub.empty:
            continue

        plt.figure(figsize=(9, 6))

        sns.lineplot(
            data=ds_sub,
            x="Epoch",
            y="GenGap",
            hue="Optimizer",
            estimator="mean",
            ci=None,
        )

        plt.axhline(0, linestyle="--", linewidth=1)
        plt.title(
            f"Generalization Gap — BS {target_bs} — {ds}"
        )
        plt.tight_layout()

        plt.savefig(
            f"plots/GenGap_{ds}_BS{target_bs}.png"
        )
        plt.close()


def plot_basin_heatmaps(df):

    for (ds, arch), sub in df.groupby(
        ["Dataset", "Architecture"]
    ):

        for opt in sub["Optimizer"].unique():

            opt_df = sub[sub["Optimizer"] == opt]

            pivot = opt_df.pivot_table(
                index="BatchSize",
                columns="LR",
                values="FinalTestAcc",
                aggfunc="mean",
            )

            if pivot.empty:
                continue

            plt.figure(figsize=(9, 7))

            sns.heatmap(
                pivot,
                annot=True,
                fmt=".2f",
                cmap="viridis",
            )

            plt.title(
                f"{opt} Basin of Success — {ds} — {arch}"
            )
            plt.xlabel("Learning Rate")
            plt.ylabel("Batch Size")

            plt.tight_layout()

            plt.savefig(
                f"basins/{opt}_{ds}_{arch}.png"
            )
            plt.close()

def summarize_tunability(df):

    rows = []

    for (ds, arch, opt), sub in df.groupby(
        ["Dataset", "Architecture", "Optimizer"]
    ):

        success = sub[~sub["Diverged"]]

        rows.append(
            {
                "Dataset": ds,
                "Architecture": arch,
                "Optimizer": opt,
                "BestAcc": success["FinalTestAcc"].max(),
                "MedianAcc": success["FinalTestAcc"].median(),
                "WorstAcc": success["FinalTestAcc"].min(),
                "SuccessRate": len(success)
                / len(sub),
                "IQR": success["FinalTestAcc"].quantile(
                    0.75
                )
                - success["FinalTestAcc"].quantile(
                    0.25
                ),
            }
        )

    summary = pd.DataFrame(rows)

    print("\n===== TUNABILITY SUMMARY =====\n")
    print(summary)

    summary.to_csv(
        "tunability_summary.csv", index=False
    )

def plot_failure_rates(df):

    rates = (
        df.groupby(["Optimizer", "BatchSize"])[
            "Diverged"
        ]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 6))

    sns.barplot(
        data=rates,
        x="BatchSize",
        y="Diverged",
        hue="Optimizer",
    )

    plt.title("Failure Probability vs Batch Size")
    plt.ylabel("Divergence Rate")
    plt.tight_layout()

    plt.savefig("plots/FailureRates.png")
    plt.close()


if __name__ == "__main__":

    df_grads, df_gen, df_final = parse_logs_for_plots(
        ROOT_DIRS
    )

    print(
        f"Gradient points: {len(df_grads)} | "
        f"GenGap points: {len(df_gen)} | "
        f"Final runs: {len(df_final)}"
    )

    plot_signal_propagation(df_grads)
    plot_gen_gap(df_gen)
    plot_basin_heatmaps(df_final)
    summarize_tunability(df_final)
    plot_failure_rates(df_final)

    print("\nAll analysis complete. Results in /plots and /basins.\n")
