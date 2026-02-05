import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ==========================================
# 1. CONFIGURATION
# ==========================================
ROOT_DIRS = ["SGD_RUNS", "ADAM_RUNS"]

# We define the order of layers to plot the "Depth" on X-axis
LAYER_ORDER = ["grad/conv1", "grad/layer1", "grad/layer2", "grad/layer3"]
LAYER_LABELS = ["Conv1", "Layer 1", "Layer 2", "Layer 3"]

# ==========================================
# 2. PARSING FUNCTION
# ==========================================
def parse_logs_for_plots(root_dirs):
    grad_data = []
    gen_data = []
    
    print("Scanning logs...")
    
    for root_dir in root_dirs:
        if not os.path.exists(root_dir): continue

        for dataset in os.listdir(root_dir):
            ds_path = os.path.join(root_dir, dataset)
            if not os.path.isdir(ds_path): continue

            for run_folder in os.listdir(ds_path):
                run_path = os.path.join(ds_path, run_folder)
                if not os.path.isdir(run_path): continue
                
                event_files = glob.glob(os.path.join(run_path, "events.out.tfevents*"))
                if not event_files: continue

                # --- Extract Params ---
                try:
                    parts = run_folder.split('_')
                    optimizer = "Adam" if "ADAM" in root_dir.upper() else "SGD"
                    arch = parts[0]
                    bs = int(next(p for p in parts if p.startswith("BS")).replace("BS", ""))
                    lr = float(next(p for p in parts if p.startswith("LR")).replace("LR", ""))
                except:
                    continue

                # --- Load TensorBoard ---
                ea = EventAccumulator(event_files[0])
                ea.Reload()
                tags = ea.Tags()['scalars']

                # 1. GET GRADIENT NORMS (At Step ~50 or 100 to catch early training)
                # We want early training because that's when gradients vanish/explode.
                steps_to_check = [0, 10, 20, 50, 100] # Try to find closest available step
                
                # Check if we have gradient tags
                if "grad/conv1" in tags:
                    # Find a common step index available in logs
                    available_steps = [s.step for s in ea.Scalars("grad/conv1")]
                    if available_steps:
                        # Pick the first non-zero step (or close to start)
                        target_step = available_steps[min(len(available_steps)-1, 5)] 
                        
                        for i, layer_tag in enumerate(LAYER_ORDER):
                            if layer_tag in tags:
                                # Get value closest to target_step
                                events = ea.Scalars(layer_tag)
                                # Simple search for value at that step
                                val = next((e.value for e in events if e.step == target_step), events[0].value)
                                
                                grad_data.append({
                                    "Optimizer": optimizer,
                                    "Architecture": arch,
                                    "Dataset": dataset,
                                    "Layer": LAYER_LABELS[i],
                                    "LayerDepth": i, # 0, 1, 2, 3
                                    "Norm": val,
                                    "BatchSize": bs,
                                    "LR": lr
                                })

                # 2. GET GENERALIZATION GAP (Series over Epochs)
                if "epoch/gen_gap" in tags:
                    events = ea.Scalars("epoch/gen_gap")
                    for e in events:
                        gen_data.append({
                            "Optimizer": optimizer,
                            "Architecture": arch,
                            "Dataset": dataset,
                            "BatchSize": bs,
                            "LR": lr,
                            "Epoch": e.step,
                            "GenGap": e.value # This is (Test - Train), so likely negative
                        })

    return pd.DataFrame(grad_data), pd.DataFrame(gen_data)

# ==========================================
# 3. PLOT: LAYER-WISE GRADIENTS
# ==========================================
def plot_signal_propagation(df):
    if df.empty: return
    
    # Filter: Only look at the Architecture Stress Test (PlainNet)
    # This is where Adam shines over SGD.
    subset = df[df['Architecture'] == 'PlainNet'] # Change to 'ResNet' to compare
    
    if subset.empty:
        print("No PlainNet gradient data found.")
        return

    datasets = subset['Dataset'].unique()
    
    for ds in datasets:
        ds_subset = subset[subset['Dataset'] == ds]
        
        plt.figure(figsize=(10, 6))
        
        # We plot the MEAN gradient norm across different LRs/Batches 
        # to show the general trend of the optimizer
        sns.lineplot(data=ds_subset, x="Layer", y="Norm", hue="Optimizer", 
                     style="Optimizer", markers=True, dashes=False, linewidth=2.5)
        
        plt.yscale("log") # CRITICAL: Gradients decay exponentially
        plt.title(f"Signal Propagation in Deep Plain Network ({ds})", fontsize=14)
        plt.ylabel("Gradient Norm (Log Scale)")
        plt.xlabel("Layer Depth (Input -> Output)")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        
        plt.savefig(f"GradientFlow_{ds}.png")
        plt.show()

# ==========================================
# 4. PLOT: GENERALIZATION GAP
# ==========================================
def plot_gen_gap(df):
    if df.empty: return

    # Focus on Large Batch (e.g., 1024 or 512) where differences are clearest
    target_bs = 1024
    subset = df[df['BatchSize'] == target_bs]
    
    # If 1024 is empty, try 512 or 128
    if subset.empty:
        target_bs = df['BatchSize'].max()
        subset = df[df['BatchSize'] == target_bs]

    datasets = subset['Dataset'].unique()
    
    for ds in datasets:
        ds_subset = subset[subset['Dataset'] == ds]
        
        # Only plot ResNet for Generalization (Standard benchmark)
        ds_subset = ds_subset[ds_subset['Architecture'] == 'ResNet']
        
        plt.figure(figsize=(10, 6))
        
        # Plot Gap vs Epoch
        sns.lineplot(data=ds_subset, x="Epoch", y="GenGap", hue="Optimizer", 
                     estimator="mean", ci=None, linewidth=2)
        
        plt.title(f"Generalization Gap (Test Acc - Train Acc) @ Batch {target_bs} - {ds}", fontsize=14)
        plt.ylabel("Gap (Negative means Overfitting)")
        plt.xlabel("Epoch")
        plt.axhline(0, color='black', linewidth=1, linestyle='--')
        plt.legend(title="Optimizer")
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f"GenGap_{ds}_BS{target_bs}.png")
        plt.show()

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    df_grads, df_gen = parse_logs_for_plots(ROOT_DIRS)
    
    print(f"\nGradient Data Points: {len(df_grads)}")
    print(f"Gen Gap Data Points: {len(df_gen)}")
    
    plot_signal_propagation(df_grads)
    plot_gen_gap(df_gen)