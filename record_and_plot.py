import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from evaluate_matrices import mean_ci

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif'],
    'font.size': 20,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 18,
    'text.usetex': False,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'axes.edgecolor': 'black',
    'axes.grid': False
})

def generate_reports_and_plots(matrics_records, visualization_records, output_dir):
    
    # record all the metrics
    df_metrics = pd.DataFrame(matrics_records)
    df_metrics["Model"] = df_metrics["Model"].replace({
        "Neural_Small": "NGCB-S",
        "Neural_Large": "NGCB-L",
        "neural_small": "NGCB-S",
        "neural_large": "NGCB-L"
    }) 
    df_metrics.to_csv(f"{output_dir}/metrics_raw.csv", index=False)
    
    summary_data = []
    for target in ["popularity", "vote_average"]:
        for model in df_metrics["Model"].unique():
            subset = df_metrics[(df_metrics["Target"] == target) & (df_metrics["Model"] == model)]
            if subset.empty: continue
            for metric in ["MSE", "RMSE", "MAE", "R2"]:
                mean_val, ci_val = mean_ci(subset[metric].values)
                summary_data.append({
                    "Target": target, "Model": model, "Metric": metric,
                    "Mean": mean_val, "CI_95": ci_val,
                    "Format_Str": f"{mean_val:.4f} ± {ci_val:.4f}"
                })
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(f"{output_dir}/metrics_summary.csv", index=False)
    
    # RMSE bar chart
    colors = ['skyblue', 'lightgreen', 'salmon', 'orange']
    unique_models = sorted(df_metrics["Model"].unique())
    model_color_map = {m: c for m, c in zip(unique_models, colors[:len(unique_models)])}

    
    for target in ["popularity", "vote_average"]:
        
        plt.figure(figsize=(8, 6)) 
        
        subset = df_summary[(df_summary["Target"] == target) & (df_summary["Metric"] == "RMSE")]

        models_plot = subset["Model"].tolist()
        means = subset["Mean"].tolist()
        cis = subset["CI_95"].tolist()
        bar_colors = [model_color_map[m] for m in models_plot]
        
        # bar chart
        bars = plt.bar(models_plot, means, yerr=cis, capsize=10, color=bar_colors, edgecolor='k', alpha=0.8)
        
        plt.title(f'RMSE Comparison: {target.capitalize()}', fontsize=14)
        plt.ylabel('RMSE', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        # add the value labels of each bar
        for bar, ci in zip(bars, cis):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + ci + 0.001 * height, 
                     f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        filename = f"{output_dir}/plot_rmse_{target}.pdf"
        plt.savefig(filename, dpi=300)
        plt.close() 

    # regression scatter plots
    seed_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    unique_targets = ["popularity", "vote_average"]
    
    for target in unique_targets:
        for model in unique_models:
            
            
            plt.figure(figsize=(6, 6))
            
           
            records = [rec for rec in visualization_records if rec["Model"] == model and rec["Target"] == target]
            
            # plot each seed's points and regression line
            for k, rec in enumerate(records):
                seed = rec["Seed"]
                yt = rec["Y_True"]
                yp = rec["Y_Pred"]
                color = seed_colors[k % len(seed_colors)]
                
                # scatter points
                plt.scatter(yt, yp, color=color, alpha=0.15, s=15)
                
                if len(yt) > 1:
                    coef = np.polyfit(yt, yp, 1)
                    poly_fn = np.poly1d(coef)
                    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
                    plt.plot(lims, poly_fn(lims), color=color, linewidth=1.5, alpha=0.6, label=f'Seed {seed}')
            
            # ideal line
            if records:
                all_yt = np.concatenate([r["Y_True"] for r in records])
                all_yp = np.concatenate([r["Y_Pred"] for r in records])
                global_lims = [min(all_yt.min(), all_yp.min()), max(all_yt.max(), all_yp.max())]
                plt.plot(global_lims, global_lims, 'k--', linewidth=2, label='Ideal', zorder=10)
            
            plt.title(f"{model} - {target.capitalize()}", fontsize=14)
            plt.xlabel("Actual Value", fontsize=12)
            plt.ylabel("Predicted Value", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize='small', loc='upper left')
            
            plt.tight_layout()
            filename = f"{output_dir}/plot_scatter_{model}_{target}.pdf"
            plt.savefig(filename, dpi=300)
            plt.close()
            print(f"Saved: {filename}")

def plot_loss_history(loss_history, output_dir):

    df_loss = pd.DataFrame(loss_history)
    plt.figure(figsize=(10, 6))
    
    # compute average loss per epoch across seeds
    # plot for each model (small and large)
    for model_name in df_loss["Model"].unique():
        subset = df_loss[df_loss["Model"] == model_name]
        avg_loss = subset.groupby("Epoch")["Loss"].mean()
        
        if model_name == "Neural_Small":
            plt.plot(avg_loss.index, avg_loss.values, linewidth=2, label="NGCB-S", color='blue')
        else:
            plt.plot(avg_loss.index, avg_loss.values, linewidth=2, label="NGCB-L", color='orange')
        
    plt.title("Neural Network Training Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    
    plt.savefig(f"{output_dir}/plot_loss_curve.pdf", dpi=300)
    plt.close()