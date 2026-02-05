"""
Adam Optimizer Analysis: Robustness and Failure Modes Investigation
--------------------------------------------------------------------
This script analyzes Adam optimizer performance across:
1. Sensitivity Analysis: Performance heatmaps across batch sizes and learning rates
2. Architectural Stress Tests: PlainNet vs ResNet comparison
3. Layer-wise Gradient Analysis: Tracking gradient norms and signal propagation
"""

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def extract_comprehensive_results(log_dir="runs/MNIST/", dataset="MNIST"):
    """
    Extract comprehensive metrics from TensorBoard logs including:
    - Training/test loss and accuracy
    - Layer-wise gradient norms
    - Convergence speed
    - Architecture type
    """
    results = []
    
    for folder in sorted(os.listdir(log_dir)):
        path = os.path.join(log_dir, folder)
        if not os.path.isdir(path): 
            continue
        
        # Parse folder name to extract hyperparameters
        # Format: MNISTPlainNet_BS32_LR0.01_SCHDLNone_Adam
        try:
            parts = folder.split('_')
            
            # Extract architecture type
            if "PlainNet" in folder:
                arch = "PlainNet"
            elif "ResNet" in folder:
                arch = "ResNet"
            else:
                arch = "Unknown"
            
            # Extract hyperparameters
            batch_size = None
            lr = None
            scheduler = None
            
            for i, part in enumerate(parts):
                if part.startswith("BS"):
                    batch_size = int(part.replace("BS", ""))
                elif part.startswith("LR"):
                    lr = float(part.replace("LR", ""))
                elif part.startswith("SCHDL"):
                    scheduler = part.replace("SCHDL", "")
            
            optim_name = parts[-1]
            
            if batch_size is None or lr is None:
                continue
                
        except Exception as e:
            print(f"Failed to parse folder {folder}: {e}")
            continue

        # Find the event file
        event_file = None
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.startswith('events.out.tfevents'):
                    event_file = os.path.join(root, file)
                    break
            if event_file:
                break
        
        if not event_file:
            print(f"No event file found in {folder}")
            continue
        
        # Load TensorBoard data
        try:
            ea = EventAccumulator(event_file)
            ea.Reload()
            
            # Get available metrics
            available_scalars = ea.Tags()['scalars']
            
            # Extract training metrics
            train_loss = None
            test_loss = None
            train_acc = None
            test_acc = None
            final_train_loss = None
            final_test_loss = None
            final_train_acc = None
            final_test_acc = None
            best_test_acc = None
            convergence_epoch = None
            min_train_loss = None
            
            # Training Loss
            if "Training/Loss" in available_scalars:
                train_loss_data = ea.Scalars("Training/Loss")
                final_train_loss = train_loss_data[-1].value
                min_train_loss = min(x.value for x in train_loss_data)
                train_loss = [x.value for x in train_loss_data]
                
                # Calculate convergence (when loss stabilizes)
                for i in range(5, len(train_loss)):
                    if train_loss[i] < 0.1:  # Threshold for "converged"
                        convergence_epoch = i
                        break
            
            # Test Loss
            if "Test/Loss" in available_scalars:
                test_loss_data = ea.Scalars("Test/Loss")
                final_test_loss = test_loss_data[-1].value
                test_loss = [x.value for x in test_loss_data]
            
            # Training Accuracy
            if "Training/Accuracy" in available_scalars:
                train_acc_data = ea.Scalars("Training/Accuracy")
                final_train_acc = train_acc_data[-1].value
                train_acc = [x.value for x in train_acc_data]
            
            # Test Accuracy
            if "Test/Accuracy" in available_scalars:
                test_acc_data = ea.Scalars("Test/Accuracy")
                final_test_acc = test_acc_data[-1].value
                best_test_acc = max(x.value for x in test_acc_data)
                test_acc = [x.value for x in test_acc_data]
                
                # Calculate convergence epoch (when reaching 95% of best accuracy)
                threshold = 0.95 * best_test_acc
                for i, acc in enumerate(test_acc):
                    if acc >= threshold:
                        convergence_epoch = i
                        break
            
            # Extract gradient norms (layer-wise)
            grad_norms = {}
            for tag in available_scalars:
                if "Gradient" in tag or "gradient" in tag:
                    grad_data = ea.Scalars(tag)
                    avg_grad_norm = np.mean([x.value for x in grad_data])
                    final_grad_norm = grad_data[-1].value
                    grad_norms[tag] = {
                        'avg': avg_grad_norm,
                        'final': final_grad_norm,
                        'max': max(x.value for x in grad_data),
                        'min': min(x.value for x in grad_data)
                    }
            
            results.append({
                "Dataset": dataset,
                "Architecture": arch,
                "Batch Size": batch_size,
                "Learning Rate": lr,
                "Scheduler": scheduler,
                "Optimizer": optim_name,
                "Final Train Loss": final_train_loss,
                "Min Train Loss": min_train_loss,
                "Final Test Loss": final_test_loss,
                "Final Train Accuracy": final_train_acc,
                "Final Test Accuracy": final_test_acc,
                "Best Test Accuracy": best_test_acc,
                "Convergence Epoch": convergence_epoch,
                "Gradient Norms": grad_norms,
                "Folder": folder
            })
            
        except Exception as e:
            print(f"Error processing {folder}: {e}")
            continue

    return pd.DataFrame(results)


def plot_sensitivity_heatmap(df, metric="Min Train Loss", arch=None, scheduler=None, invert=False):
    """
    Generate heatmap showing 'basin of success' across hyperparameters.
    For loss metrics, lower is better; for accuracy, higher is better.
    """
    filtered_df = df.copy()
    
    if arch:
        filtered_df = filtered_df[filtered_df["Architecture"] == arch]
    if scheduler:
        filtered_df = filtered_df[filtered_df["Scheduler"] == scheduler]
    
    # Remove NaN values
    filtered_df = filtered_df.dropna(subset=[metric])
    
    if len(filtered_df) == 0:
        print(f"No data available for {metric} with filters: arch={arch}, scheduler={scheduler}")
        return None
    
    # Create pivot table
    pivot = filtered_df.pivot_table(
        values=metric,
        index="Learning Rate",
        columns="Batch Size",
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    
    # For loss metrics, invert colormap (lower is better = greener)
    if "Loss" in metric:
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn_r', 
                    cbar_kws={'label': metric})
    else:
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', 
                    cbar_kws={'label': metric}, vmin=0, vmax=1)
    
    title = f"{metric} - Adam Optimizer\n"
    if arch:
        title += f"Architecture: {arch}"
    if scheduler:
        title += f" | Scheduler: {scheduler}"
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.tight_layout()
    
    return plt.gcf()


def plot_architecture_comparison(df, metric="Min Train Loss"):
    """
    Compare PlainNet vs ResNet to understand architectural robustness.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, arch in enumerate(['PlainNet', 'ResNet']):
        arch_df = df[df['Architecture'] == arch]
        arch_df = arch_df.dropna(subset=[metric])
        
        if len(arch_df) == 0:
            axes[idx].text(0.5, 0.5, f'No data for {arch}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f"{arch} - {metric}", fontsize=12, fontweight='bold')
            continue
        
        pivot = arch_df.pivot_table(
            values=metric,
            index="Learning Rate",
            columns="Batch Size",
            aggfunc='mean'
        )
        
        # For loss metrics, invert colormap
        if "Loss" in metric:
            sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn_r', 
                       ax=axes[idx], cbar_kws={'label': metric})
        else:
            sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', 
                       ax=axes[idx], cbar_kws={'label': metric}, vmin=0, vmax=1)
        
        axes[idx].set_title(f"{arch} - {metric}", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel("Batch Size")
        axes[idx].set_ylabel("Learning Rate")
    
    plt.suptitle("Architectural Stress Test: PlainNet vs ResNet with Adam", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def analyze_gradient_propagation(df):
    """
    Analyze layer-wise gradient norms to understand signal propagation.
    """
    print("\n" + "="*70)
    print("GRADIENT PROPAGATION ANALYSIS")
    print("="*70)
    
    for arch in df['Architecture'].unique():
        arch_df = df[df['Architecture'] == arch]
        
        print(f"\n{arch} Architecture:")
        print("-" * 70)
        
        # Aggregate gradient norms across runs
        all_grad_norms = []
        for idx, row in arch_df.iterrows():
            if row['Gradient Norms']:
                all_grad_norms.append(row['Gradient Norms'])
        
        if all_grad_norms:
            # Analyze gradient statistics
            print(f"  Number of runs analyzed: {len(all_grad_norms)}")
            
            # Aggregate by layer
            layer_stats = {}
            for grad_norm_dict in all_grad_norms:
                for layer_name, stats in grad_norm_dict.items():
                    if layer_name not in layer_stats:
                        layer_stats[layer_name] = {'avg': [], 'max': [], 'min': []}
                    layer_stats[layer_name]['avg'].append(stats['avg'])
                    layer_stats[layer_name]['max'].append(stats['max'])
                    layer_stats[layer_name]['min'].append(stats['min'])
            
            print(f"\n  Layer-wise Gradient Statistics:")
            for layer, stats in sorted(layer_stats.items()):
                avg_grad = np.mean(stats['avg'])
                max_grad = np.mean(stats['max'])
                print(f"    {layer:50s} | Avg: {avg_grad:8.6f} | Max: {max_grad:8.6f}")


def generate_performance_summary(df):
    """
    Generate comprehensive performance summary.
    """
    print("\n" + "="*70)
    print("ADAM OPTIMIZER PERFORMANCE SUMMARY")
    print("="*70)
    
    for arch in df['Architecture'].unique():
        arch_df = df[df['Architecture'] == arch]
        
        print(f"\n{arch} Architecture:")
        print("-" * 70)
        
        # Best performing configuration (lowest loss)
        metric_col = 'Min Train Loss'
        if metric_col in arch_df.columns:
            arch_df_clean = arch_df.dropna(subset=[metric_col])
            if len(arch_df_clean) > 0:
                best_run = arch_df_clean.loc[arch_df_clean[metric_col].idxmin()]
                print(f"  Best Configuration (Lowest Loss):")
                print(f"    Batch Size: {best_run['Batch Size']}")
                print(f"    Learning Rate: {best_run['Learning Rate']}")
                print(f"    Scheduler: {best_run['Scheduler']}")
                print(f"    Min Train Loss: {best_run[metric_col]:.6f}")
                print(f"    Final Train Loss: {best_run['Final Train Loss']:.6f}")
                if best_run['Convergence Epoch'] is not None:
                    print(f"    Convergence Epoch: {best_run['Convergence Epoch']}")
        
        # Basin of success analysis (runs with loss < 0.1)
        print(f"\n  Hyperparameter Robustness:")
        if 'Min Train Loss' in arch_df.columns:
            arch_df_clean = arch_df.dropna(subset=['Min Train Loss'])
            good_runs = arch_df_clean[arch_df_clean['Min Train Loss'] < 0.1]
            print(f"    Runs achieving <0.1 loss: {len(good_runs)}/{len(arch_df_clean)}")
            
            if len(good_runs) > 0:
                print(f"    Learning Rate range: {good_runs['Learning Rate'].min():.4f} - {good_runs['Learning Rate'].max():.4f}")
                print(f"    Batch Size range: {int(good_runs['Batch Size'].min())} - {int(good_runs['Batch Size'].max())}")
        
        # Convergence speed
        if 'Convergence Epoch' in arch_df.columns:
            converged = arch_df.dropna(subset=['Convergence Epoch'])
            if len(converged) > 0:
                avg_convergence = converged['Convergence Epoch'].mean()
                print(f"    Average convergence epoch: {avg_convergence:.1f}")
        
        # Learning rate sensitivity
        print(f"\n  Learning Rate Sensitivity:")
        if 'Learning Rate' in arch_df.columns and 'Min Train Loss' in arch_df.columns:
            for lr in sorted(arch_df['Learning Rate'].unique()):
                lr_runs = arch_df[arch_df['Learning Rate'] == lr].dropna(subset=['Min Train Loss'])
                if len(lr_runs) > 0:
                    avg_loss = lr_runs['Min Train Loss'].mean()
                    std_loss = lr_runs['Min Train Loss'].std()
                    print(f"    LR={lr:.4f}: Avg Loss={avg_loss:.6f} (±{std_loss:.6f})")


def save_results(df, output_dir="analysis_resultsCIFAR10"):
    """
    Save analysis results and plots.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Save raw data
    df_export = df.copy()
    # Convert gradient norms dict to string for CSV export
    df_export['Gradient Norms'] = df_export['Gradient Norms'].apply(str)
    df_export.to_csv(f"{output_dir}/adam_analysis_raw_data.csv", index=False)
    print(f"\n✓ Raw data saved to {output_dir}/adam_analysis_raw_data.csv")
    
    # Generate and save heatmaps for each architecture and metric
    metrics = ['Min Train Loss', 'Final Train Loss']
    
    for arch in df['Architecture'].unique():
        for scheduler in df['Scheduler'].unique():
            for metric in metrics:
                fig = plot_sensitivity_heatmap(df, metric=metric, arch=arch, scheduler=scheduler)
                if fig:
                    filename = f"{output_dir}/heatmap_{arch}_{scheduler}_{metric.replace(' ', '_')}.png"
                    fig.savefig(filename, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"✓ Saved heatmap: {filename}")
    
    # Architecture comparison
    for metric in metrics:
        fig = plot_architecture_comparison(df, metric=metric)
        if fig:
            fig.savefig(f"{output_dir}/architecture_comparison_{metric.replace(' ', '_')}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ Saved architecture comparison: {output_dir}/architecture_comparison_{metric.replace(' ', '_')}.png")


def main():
    """
    Main analysis pipeline for Adam optimizer investigation.
    """
    print("="*70)
    print("ADAM OPTIMIZER ANALYSIS: Robustness and Failure Modes")
    print("="*70)
    
    # Extract results from MNIST runs
    print("\nExtracting data from MNIST runs...")
    df_mnist = extract_comprehensive_results("runs2", dataset="CIFAR10")
    print(f"✓ Loaded {len(df_mnist)} MNIST runs")
    
    # Check for CIFAR10 runs
    if os.path.exists("runs/CIFAR10/"):
        print("\nExtracting data from CIFAR10 runs...")
        df_cifar10 = extract_comprehensive_results("runs/CIFAR10/", dataset="CIFAR10")
        print(f"✓ Loaded {len(df_cifar10)} CIFAR10 runs")
        df = pd.concat([df_mnist, df_cifar10], ignore_index=True)
    else:
        df = df_mnist
    
    if len(df) == 0:
        print("No data found!")
        return
    
    # Generate comprehensive analysis
    generate_performance_summary(df)
    analyze_gradient_propagation(df)
    
    # Save all results
    save_results(df)
    
    print("\n" + "="*70)
    print("Analysis complete! Check 'analysis_results' directory for outputs.")
    print("="*70)


if __name__ == "__main__":
    main()