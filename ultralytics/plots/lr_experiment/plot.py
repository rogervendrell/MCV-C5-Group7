import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics_from_csv(csv_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    df = df.sort_values('lr')
    
    # Plot mAP metrics separately
    for metric in ['mAP50', 'mAP50-95']:
        plt.figure(figsize=(4,3))
        plt.plot(df['lr'], df[metric], marker='o', linestyle='--', color='orange')
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel(metric)
        plt.title(f'{metric} vs Learning Rate')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(0, 1) 
        
        filename = os.path.join(save_dir, f'{metric}_vs_lr.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Precision vs Recall
    plt.figure(figsize=(4,3))
    plt.plot(df['precision'], df['recall'], marker='o', color="orange", linestyle="--")
    
    for i, lr in enumerate(df['lr']):
        plt.annotate(f'lr={lr}', 
                     (df['precision'].iloc[i], df['recall'].iloc[i]),
                     rotation=0,
                     ha='left', 
                     va='bottom',
                     fontsize=8)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("Precision vs Recall")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    filename = os.path.join(save_dir, 'precision_vs_recall.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved in {save_dir}")


csv_file = "/ghome/group07/MCV-C5-Group7/ultralytics/output/task_e/runs/detect/task_e_experiment/reduced.csv"
output_dir = "/ghome/group07/MCV-C5-Group7/ultralytics/plots/lr_experiment/plots"
plot_metrics_from_csv(csv_file, output_dir)