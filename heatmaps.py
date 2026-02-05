import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Update these if your folders have different names
ROOT_DIRS = ["SGD_RUNS", "ADAM_RUNS"] 

# The tag used for Test Accuracy in your TensorBoard logs
# The script checks this list until it finds a match.
TAGS_TO_CHECK = [
    'Accuracy/Test', 
    'Test/Accuracy', 
    'Test_Acc', 
    'TestAccuracy', 
    'epoch/test_acc',  # Matches your specific logging code provided earlier
    'Accuracy/Test_Acc'
]

# ==========================================
# 2. PARSING FUNCTION
# ==========================================
def parse_tensorboard_logs(root_dirs):
    data = []
    print(f"Scanning directories: {root_dirs}...")
    
    for root_dir in root_dirs:
        if not os.path.exists(root_dir):
            print(f"Warning: Directory '{root_dir}' not found.")
            continue

        # Level 1: Dataset (e.g., CIFAR10, MNIST)
        for dataset_name in os.listdir(root_dir):
            dataset_path = os.path.join(root_dir, dataset_name)
            if not os.path.isdir(dataset_path): continue
                
            # Level 2: Run Folder (e.g., ResNet_BS128_LR0.001_Cosine)
            for run_folder in os.listdir(dataset_path):
                run_path = os.path.join(dataset_path, run_folder)
                if not os.path.isdir(run_path): continue
                
                # Check for TensorBoard event file
                event_files = glob.glob(os.path.join(run_path, "events.out.tfevents*"))
                if not event_files: continue
                
                # --- PARSE FILENAME FOR HYPERPARAMETERS ---
                try:
                    # Expected format: {Arch}_BS{Batch}_LR{LR}_{Scheduler}
                    parts = run_folder.split('_')
                    
                    optimizer = "Adam" if "ADAM" in root_dir.upper() else "SGD"
                    arch = parts[0]
                    
                    # Extract Batch Size
                    bs_str = next(p for p in parts if p.startswith("BS"))
                    batch_size = int(bs_str.replace("BS", ""))
                    
                    # Extract Learning Rate
                    lr_str = next(p for p in parts if p.startswith("LR"))
                    lr = float(lr_str.replace("LR", ""))
                    
                    # Extract Scheduler (Optional, for debugging)
                    lr_index = parts.index(lr_str)
                    if len(parts) > lr_index + 1:
                        scheduler = "_".join(parts[lr_index+1:])
                    else:
                        scheduler = "None"
                        
                except Exception as e:
                    # Skip folders that don't match the naming convention
                    # print(f"Skipping {run_folder}: {e}")
                    continue

                # --- LOAD DATA FROM TENSORBOARD ---
                try:
                    ea = EventAccumulator(event_files[0])
                    ea.Reload()
                    available_tags = ea.Tags()['scalars']
                    
                    # Find the correct accuracy tag
                    found_tag = None
                    for tag in TAGS_TO_CHECK:
                        if tag in available_tags:
                            found_tag = tag
                            break
                    
                    if found_tag:
                        # Get the FINAL value logged
                        final_acc = ea.Scalars(found_tag)[-1].value
                        
                        data.append({
                            "Optimizer": optimizer,
                            "Dataset": dataset_name,
                            "Architecture": arch,
                            "BatchSize": batch_size,
                            "LR": lr,
                            "Scheduler": scheduler,
                            "TestAccuracy": final_acc
                        })
                except Exception as e:
                    print(f"Error reading log {run_folder}: {e}")

    return pd.DataFrame(data)

# ==========================================
# 3. PLOTTING FUNCTION
# ==========================================
def plot_heatmaps(df):
    if df.empty:
        print("No data found! Check your folder paths or tags.")
        return

    datasets = df['Dataset'].unique()
    architectures = df['Architecture'].unique()

    for ds in datasets:
        for arch in architectures:
            # Filter data for this specific chart
            subset = df[(df['Dataset'] == ds) & (df['Architecture'] == arch)]
            
            if subset.empty:
                continue
                
            print(f"Generating Heatmap for {ds} - {arch}...")
            
            # Create a figure with 2 subplots (Side by Side)
            fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
            
            # Determine color scale range (e.g., 10% to Max achieved)
            # This ensures both plots share the same colors for fair comparison
            vmin = 0.1 
            vmax = subset['TestAccuracy'].max()

            optimizers = ['Adam', 'SGD']
            
            for i, optim in enumerate(optimizers):
                opt_data = subset[subset['Optimizer'] == optim]
                
                if opt_data.empty:
                    axes[i].text(0.5, 0.5, f"No Data for {optim}", 
                                 ha='center', va='center', fontsize=12)
                    axes[i].set_title(f"{optim}")
                    continue
                
                # --- AGGREGATION LOGIC (CRITICAL STEP) ---
                # We pivot the table: Rows=BatchSize, Cols=LR.
                # aggfunc='max' ensures that if we have multiple schedulers, 
                # we pick the BEST result for that cell.
                pivot_table = opt_data.pivot_table(
                    index="BatchSize", 
                    columns="LR", 
                    values="TestAccuracy", 
                    aggfunc='max'
                )
                
                # Sort Batch Sizes (smallest to largest)
                pivot_table = pivot_table.sort_index(ascending=True)
                
                # Plot
                sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="viridis", 
                            ax=axes[i], vmin=vmin, vmax=vmax, cbar=True,
                            linewidths=0.5, linecolor='gray')
                
                axes[i].set_title(f"{optim} (Peak Accuracy)", fontsize=14, fontweight='bold')
                axes[i].set_ylabel("Batch Size", fontsize=12)
                axes[i].set_xlabel("Learning Rate", fontsize=12)

            # Final Layout Adjustments
            plt.suptitle(f"Basin of Success: {ds} ({arch})", fontsize=18)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Make room for suptitle
            
            # Save File
            filename = f"Heatmap_{ds}_{arch}.png"
            plt.savefig(filename, dpi=300)
            print(f"Saved: {filename}")
            plt.close() # Free memory

# ==========================================
# 4. EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # 1. Parse Data
    df = parse_tensorboard_logs(ROOT_DIRS)
    
    # 2. Quality Check
    if not df.empty:
        print(f"\nSuccessfully extracted {len(df)} runs.")
        print(df.head())
        
        # Optional: Save the raw data to CSV for your report
        df.to_csv("heatmap_data_raw.csv", index=False)
        print("Raw data saved to 'heatmap_data_raw.csv'")
        
        # 3. Generate Plots
        plot_heatmaps(df)
    else:
        print("DataFrame is empty. No plots generated.")