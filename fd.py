import os
import glob
from collections import defaultdict

# ==========================================
# CONFIGURATION
# ==========================================
ROOT_DIRS = ["SGD_RUNS", "ADAM_RUNS"]  # Update if your folders are named differently

def find_duplicates(root_dirs):
    # Dictionary to store paths for each configuration
    # Key: Tuple (Optimizer, Dataset, Architecture, BatchSize, LR)
    # Value: List of file paths
    runs_map = defaultdict(list)
    
    print(f"Scanning directories: {root_dirs}...\n")
    
    for root_dir in root_dirs:
        if not os.path.exists(root_dir):
            print(f"Warning: Directory '{root_dir}' not found.")
            continue

        # Walk structure: Root -> Dataset -> Run_Folder
        for dataset_name in os.listdir(root_dir):
            dataset_path = os.path.join(root_dir, dataset_name)
            
            if not os.path.isdir(dataset_path):
                continue
                
            for run_folder in os.listdir(dataset_path):
                run_path = os.path.join(dataset_path, run_folder)
                
                if not os.path.isdir(run_path):
                    continue
                
                # Check for tfevents file to ensure it's a valid run
                if not glob.glob(os.path.join(run_path, "events.out.tfevents*")):
                    continue
                
                # --- PARSE HYPERPARAMETERS ---
                try:
                    parts = run_folder.split('_')
                    
                    optimizer = "Adam" if "ADAM" in root_dir.upper() else "SGD"
                    arch = parts[0]
                    bs_str = next(p for p in parts if p.startswith("BS"))
                    batch_size = int(bs_str.replace("BS", ""))
                    lr_str = next(p for p in parts if p.startswith("LR"))
                    lr = float(lr_str.replace("LR", ""))
                    
                    # Create a unique signature for this experiment
                    # Note: We include Schedule (parts[-1]) if you want to treat different schedules as different runs.
                    # If you want to find duplicates IGNORING schedule, remove it from this tuple.
                    config_key = (optimizer, dataset_name, arch, batch_size, lr)
                    
                    runs_map[config_key].append(run_path)
                    
                except Exception as e:
                    # print(f"Skipping {run_folder}: {e}")
                    continue

    # --- REPORT DUPLICATES ---
    duplicate_count = 0
    print("="*60)
    print("DUPLICATE RUNS FOUND")
    print("="*60)
    
    for config, paths in runs_map.items():
        if len(paths) > 1:
            duplicate_count += 1
            optimizer, dataset, arch, bs, lr = config
            print(f"\n[DUPLICATE] {optimizer} | {dataset} | {arch} | BS: {bs} | LR: {lr}")
            print(f"Found {len(paths)} folders:")
            for p in paths:
                print(f"  - {p}")

    if duplicate_count == 0:
        print("\nNo duplicates found! Your data is clean.")
    else:
        print(f"\nTotal duplicate configurations found: {duplicate_count}")

if __name__ == "__main__":
    find_duplicates(ROOT_DIRS)