import os
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import clone

from data_prepocessing import preprocess_data
from evaluate_matrices import mse_rmse_mae_r2
from neural_mode import MovieDataset, collate_fn, NeuralCB
from record_and_plot import generate_reports_and_plots, plot_loss_history

import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader

# the random seeds for all the simulations and the data file
SEEDS = [42, 7, 21, 84, 63]
OUTPUT_DIR = "project_results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
matrics_records = []
visualization_records = []

def linear_regression_models(md_linear, seed):
    # convert polars dataframe to pandas dataframe for sklearn
    md_linear = md_linear.to_pandas()
    X_linear = md_linear.drop(columns=["popularity", "vote_average"]).values
    Y_linear = md_linear[["popularity", "vote_average"]].values # target variable for the porpularitiy and vote average

    # linear model define
    linear_model = {
        "Ridge": MultiOutputRegressor(Ridge(alpha = 1.0)),  # Placeholder for Ridge regression
        "LASSO": MultiOutputRegressor(make_pipeline(StandardScaler(), Lasso(alpha = 0.1)))   # Placeholder for LASSO regression
    }
    
    for seed in SEEDS:
        # split the data
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_linear, Y_linear, test_size=0.2, random_state=seed
        )

        for model_name, model in linear_model.items():
            clone_model = clone(model)
            clone_model.fit(X_train, Y_train)
            y_pred = clone_model.predict(X_test)
            
            # evaluate the model
            for i in range(2):
                target = "popularity" if i == 0 else "vote_average"
                metrics = mse_rmse_mae_r2(Y_test[:, i], y_pred[:, i])

                # store all the metrics
                matrics_records.append({
                    "Model": model_name,
                    "Target": target,
                    "Seed": seed,
                    **metrics
                })

                # store the visualization records
                visualization_records.append({
                    "Model": model_name,
                    "Target": target,
                    "Seed": seed,
                    "Y_True": Y_test[:, i],
                    "Y_Pred": y_pred[:, i]
                })

def neural_regression(md, md_size, model_type, seeds):

    # use gpu to accelerate training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    md_neural, md_neural_size = md, md_size
    md_neural = md_neural.to_pandas()
    loss_history = []
    
    print(f"Starting Neural Regression ({model_type}) with {len(seeds)} seeds")
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        indices = np.arange(len(md_neural))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=seed)
        
        train_ds = MovieDataset(md_neural.iloc[train_idx])
        test_ds = MovieDataset(md_neural.iloc[test_idx])
        
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers = 0)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers = 0)
        
        # model, optimizer, loss function
        model = NeuralCB(md_neural_size, embed_dim=16, model_type=model_type).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_function = nn.MSELoss()
        
        # training loop
        for epoch in range(500):
            model.train()
            loss_init = 0
            for batch in train_loader:
                r, p, c, g, l, y = [item.to(device) for item in batch]
                optimizer.zero_grad()
                preds, w_pop, w_vote = model(r, p, c, g, l)
                loss = loss_function(preds, y) + 0.01 * torch.mean(w_pop ** 2) + torch.mean(w_vote ** 2)
                loss.backward()
                optimizer.step()
                loss_init += loss.item()

                if loss < 0.001:
                    break
                
            if epoch % 10 == 0:
                print(f"Neural_{model_type} Seed {seed} Epoch {epoch+1}/500 Loss: {loss_init/len(train_loader):.4f}")
            
            loss_history.append({
                "Model": f"Neural_{model_type}",
                "Seed": seed,
                "Epoch": epoch + 1,
                "Loss": loss_init / len(train_loader)
            })
                
        # evaluation
        model.eval()
        preds_list, true_list = [], []
        with torch.no_grad():
            for batch in test_loader:
                r, p, c, g, l, y = [item.to(device) for item in batch]
                preds, _, _ = model(r, p, c, g, l)
                preds_list.append(preds.cpu())
                true_list.append(y.cpu())
        
        preds = torch.cat(preds_list).numpy()
        targets = torch.cat(true_list).numpy()
        
        # compute metrics
        for i in range(2):
            target = "popularity" if i == 0 else "vote_average"
            metrics = mse_rmse_mae_r2(targets[:, i], preds[:, i])

            matrics_records.append({
                "Model": f"Neural_{model_type}",
                "Target": target,
                "Seed": seed,
                **metrics
            })

            visualization_records.append({
                "Model": f"Neural_{model_type}",
                "Target": target,
                "Seed": seed,
                "Y_True": targets[:, i],
                "Y_Pred": preds[:, i]
            })


    return loss_history

def main():
    # preprocess the data
    md_linear, md_neural, md_neural_size = preprocess_data()
    
    # begin training and evaluation for traitional linear regression （Ridge and LASSO)
    print("Begin Linear Regression Models (Ridge and LASSO)")
    linear_regression_models(md_linear, SEEDS)

    # begin training and evaluation for neural network regression (Small and Large)
    print("Begin Neural Network Regression Models (Small and Large)")
    loss_history_small = neural_regression(md_neural, md_neural_size, model_type="Small", seeds=SEEDS)
    loss_history_large = neural_regression(md_neural, md_neural_size, model_type="Large", seeds=SEEDS)
    loss_history = loss_history_small + loss_history_large

    # plot loss history
    plot_loss_history(loss_history, OUTPUT_DIR)

    # reord and plot all the results
    generate_reports_and_plots(matrics_records, visualization_records, OUTPUT_DIR)

    print("All tasks completed.")

if __name__ == "__main__":
    main()