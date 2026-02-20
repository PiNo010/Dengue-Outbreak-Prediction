import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score ,mean_absolute_error ,explained_variance_score,confusion_matrix,accuracy_score,classification_report
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.model_selection import cross_validate,GroupKFold,KFold
from sklearn.linear_model import BayesianRidge
from catboost import CatBoostRegressor 
from catboost import Pool, cv
from CustomStackRegressor import StackRegressor
from StackingRegressorCountry import StackRegressorCountry
from train import train_model
import seaborn as sns
from StackRegressor_NN import StackRegressor_NN
from Model import MLP
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import joblib

def plot_metrics(y_test,y_pred,country = None,model_name = None):
    # Valuta il modello
    mse = mean_squared_error(y_test,  y_pred)
    r2 = r2_score(y_test,  y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    
    
 
    print(f"Explained Variance Score: {evs:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R^2 Score: {r2:.2f}')
    
    
    # corr_matrix = X.drop(columns=["country","date"]).corr()

    # # Plot della heatmap
    # plt.figure()
    # sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, linewidths=0.5)
    # plt.title("Matrice di correlazione tra feature")

    
    
    # importances = model.feature_importances_
    # sorted_idx = importances.argsort()
    # plt.figure(figsize=(10, 6))
    # sns.barplot(x=importances[sorted_idx], y=X_train.columns[sorted_idx])
    # plt.xlabel("Importanza")
    # plt.ylabel("Feature")
    # plt.title("Feature Importance")


    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(y_test)), y_test, label='Real', linewidth=2)
    plt.plot(np.arange(len(y_pred)), y_pred, label='Predicted', linewidth=1)
    plt.xlabel("Month")
    plt.ylabel("Cases (log(incidence))")
    plt.title(f"Plot of {country} - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./plots/{country}_{model_name}_prediction.png', dpi=300, bbox_inches='tight')

    # plt.figure(figsize=(10, 6))
    # plt.scatter(np.arange(len(y_test)), y_test, label='Reali')
    
    # plt.scatter(np.arange(len(y_pred)), y_pred, label='Predetti',alpha=0.6)
    # plt.vlines(x=np.arange(len(y_test)), ymin=0, ymax=y_test, color='gray', alpha=0.3, linewidth=0.8)
    # plt.grid(True)
    plt.show()
    
    return {"country":country,'MSE':mse,'mae':mae,'R2':r2,'evs':evs}
     

def LabelEncord_categorical(df):
    df = pd.get_dummies(df, columns=['country'], drop_first=True,dtype=int)
    return df
   

def dataset_pre_processing(encode_flag):
    dataset = pd.read_csv("./with economics dataset.csv")
    dataset = dataset[ dataset['cases'] >= 0]
   
    dataset['cases'] = dataset["cases"] / dataset["Population, total"] *100_000
    # dataset = dataset[ dataset['cases'] < 150]
    dataset['cases'] = np.log1p(dataset["cases"])
    
    dataset['year'] = dataset['date'].apply(lambda x: int(x[:4]))
    dataset['month'] = dataset['date'].apply(lambda x: int(x[5:7]))
    


    # dataset['lagged_1_month'] = dataset.groupby('country')['cases'].shift(1)
    # dataset['lagged_3_month'] = dataset.groupby('country')['cases'].shift(3)
    # dataset['lagged_6_month'] = dataset.groupby('country')['cases'].shift(6)
    dataset['lagged_1_year'] = dataset.groupby('country')['cases'].shift(12)
    # dataset['lagged_2_year'] = dataset.groupby('country')['cases'].shift(24)
    # dataset['lagged_3_year'] = dataset.groupby('country')['cases'].shift(36)
    
    
    # dataset['rolling_3_month'] = dataset.groupby('country')['cases'].shift(1).rolling(3,min_periods=1).mean()
    # dataset['rolling_6_month'] = dataset.groupby('country')['cases'].shift(1).rolling(6,min_periods=1).mean()
    # dataset['rolling_1_year'] = dataset.groupby('country')['cases'].shift(1).rolling(12,min_periods=1).mean()
    
    # dataset['std_3_month'] = dataset.groupby('country')['cases'].shift(1).rolling(3,min_periods=1).std()
    # dataset['std_6_month'] = dataset.groupby('country')['cases'].shift(1).rolling(6,min_periods=1).std()
    # dataset['std_3_month'] = dataset.groupby('country')['cases'].shift(1).rolling(3,min_periods=1).std()
    # dataset['std_1_year'] = dataset.groupby('country')['cases'].shift(1).rolling(12,min_periods=1).std()
    
    dataset = dataset.fillna(0)
   
    # corr_matrix = dataset.drop(columns=["country","date"]).corr()

    # # Plot della heatmap
    # plt.figure()
    # sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, linewidths=0.5)
    # plt.title("Matrice di correlazione tra feature")
    # plt.show()
    

    
    if encode_flag:
        dataset = LabelEncord_categorical(dataset)
    
    print("Incidence: ", dataset['cases'].mean())
   
    y = dataset['cases']
    
    X = dataset.drop(columns=['date','cases','deaths','confirmed_cases','severe_cases','cfr_ci_lower', 'd2m','lai_lv','lai_hv',
                              'People using at least basic drinking water services, urban (% of urban population)',
                              'People using safely managed drinking water services, urban (% of urban population)',
                              'cfr','prop_sev','cfr_ci_upper','prop_sev_ci_lower','prop_sev_ci_upper','Population, total',
                              'People using at least basic drinking water services (% of population)','Average precipitation in depth (mm per year)',
                              'cl'])  
    
    # -------- SAMPLE FREQUENCIES-------------
    # plt.figure(figsize=(10, 6))
    # sns.histplot(y, bins=30, kde=False, color='blue', alpha=0.6)
    # plt.title('Distribuzione di y (Target)', fontsize=15)
    # plt.xlabel('Valore di y', fontsize=12)
    # plt.ylabel('Frequenza', fontsize=12)
    # plt.grid(True)
    # plt.show()

    return X,y


def main():
   
    X,y = dataset_pre_processing(False)
    
    
    X_train, X_test, y_train, y_test = X[ X['year'] < 2024 ] , X[ X['year'] >= 2024 ] ,y[ X['year'] < 2024 ], y[ X['year'] >= 2024 ]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    cat_features = ['country']
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    test_pool  = Pool(X_test,  y_test,  cat_features=cat_features)
    
    model = CatBoostRegressor(iterations=300,
                              border_count=110,
                              depth=8,
                              learning_rate=0.05,
                              l2_leaf_reg=7,
                              eval_metric="RMSE",
                              loss_function='MAE',
                              verbose=0)
 
    model.fit(train_pool, eval_set=test_pool)
    countries = X_test['country'].unique()
    results = []
    for country in countries:
        print("COUNTRY: ",country)
        y_pred = model.predict(X_test[ X_test['country'] == country ])
        y_test_new = y_test[ X_test['country'] == country ]
        results.append(plot_metrics(y_test_new,y_pred,country))
        print()
    return results
        

def ensemble():
    X,y = dataset_pre_processing(False)
    print(X.columns)
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    X_train, X_test, y_train, y_test = X[ X['year'] < 2024 ] , X[ X['year'] >= 2024 ] ,y[ X['year'] < 2024 ], y[ X['year'] >= 2024 ]
    
    cat_features = ["country"]
    
    meta_model = CatBoostRegressor(iterations=600,
                                   verbose = 0,
                                   eval_metric='RMSE',
                                   loss_function='MAE',
                                   l2_leaf_reg=8,
                                   learning_rate=0.05,
                                   depth=10)

    cat_model = CatBoostRegressor(iterations=400,
                              border_count=110,
                              depth=8,
                              learning_rate=0.05,
                              l2_leaf_reg=7,
                              eval_metric="RMSE",
                              loss_function='MAE',
                              verbose = 0)
    

    cat_model_2 = CatBoostRegressor(iterations=500,
                              border_count=90,
                              eval_metric="RMSE",
                              learning_rate=0.01,
                              l2_leaf_reg=10,
                              loss_function='MAE',
                              verbose = 0)
    
    cat_model_3 = CatBoostRegressor(iterations=500,
                              eval_metric="MAE",
                              loss_function='RMSE',
                              verbose = 0)
    
    cat_model_4 = CatBoostRegressor(iterations=600,
                              eval_metric="RMSE",
                              loss_function='MAE',
                              learning_rate=0.05,
                              verbose = 0,
                              l2_leaf_reg=10,
                              )
    
    cat_model_5 = CatBoostRegressor(iterations=300,
                              eval_metric="RMSE",
                              loss_function='MAE',
                              verbose = 0
                            )
    
    input_models = [cat_model,cat_model_2,cat_model_3,cat_model_4,cat_model_5]
    
    
  
    
    stacked_model = StackRegressor(input_models,
                           meta_model=meta_model,
                           cat_features=cat_features)
    
  
    stacked_model.fit(X_train,y_train)
    joblib.dump(stacked_model, 'stacked_model.pkl')
    

    # PREDIZIONE
    countries = X_test['country'].unique()
    results = []
    for country in countries:
        print("COUNTRY: ",country)
        y_pred =stacked_model.predict(X_test[ X_test['country'] == country ])
        y_test_new = y_test[ X_test['country'] == country ]
        results.append(plot_metrics(y_test_new,y_pred,country,model_name="Stacked Model"))
        print()
    return results


def ensemble_NN():
    X,y = dataset_pre_processing(False)
    # print(X.columns)
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    X_train, X_test, y_train, y_test = X[ X['year'] < 2024 ] , X[ X['year'] >= 2024 ] ,y[ X['year'] < 2024 ], y[ X['year'] >= 2024 ]
    
    cat_features = ["country"]

    cat_model = CatBoostRegressor(iterations=400,
                              border_count=110,
                              depth=8,
                              learning_rate=0.05,
                              l2_leaf_reg=7,
                              eval_metric="RMSE",
                              loss_function='MAE',
                              verbose = 0)
    

    cat_model_2 = CatBoostRegressor(iterations=500,
                              border_count=110,
                              eval_metric="RMSE",
                              learning_rate=0.01,
                              l2_leaf_reg=10,
                              depth = 4,
                              loss_function='MAE',
                              verbose = 0)
    
    cat_model_3 = CatBoostRegressor(iterations=300,
                              border_count=110,
                              depth=8,
                              learning_rate=0.05,
                              l2_leaf_reg=7,
                              eval_metric="RMSE",
                              loss_function='MAE',
                              verbose=0)
    
    cat_model_4 = CatBoostRegressor(iterations=600,
                              eval_metric="RMSE",
                              loss_function='MAE',
                              learning_rate=0.05,
                              verbose = 0,
                              l2_leaf_reg=10,
                              depth=4)
    
    cat_model_5 = CatBoostRegressor(iterations=300,
                              eval_metric="MAE",
                              loss_function='RMSE',
                              l2_leaf_reg=6,
                              verbose = 0
                            )
    

    
    input_models = [cat_model,cat_model_2,cat_model_3,cat_model_4,cat_model_5]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    meta_model = MLP( [5] + [32] + [64]*2 + [1],nn.ReLU ).to(device)
    
    stacked_model = StackRegressor_NN(input_models,
                           meta_model=meta_model,
                           cat_features=cat_features)
    
  
    stacked_model.fit(X_train,y_train)

    # PREDIZIONE
    countries = X_test['country'].unique()
    results = []
    for country in countries:
        print("COUNTRY: ",country)
        y_pred =stacked_model.predict(X_test[ X_test['country'] == country ])
        y_test_new = y_test[ X_test['country'] == country ]
        results.append(plot_metrics(y_test_new,y_pred,country))
        print()
    return results


def neural_network():
    from torch.utils.data import DataLoader, TensorDataset
    
    X,y = dataset_pre_processing(False)
    
    X_train, X_test, y_train, y_test = X[ X['year'] < 2024 ] , X[ X['year'] >= 2024 ] ,y[ X['year'] < 2024 ], y[ X['year'] >= 2024 ]
    
    # NORMALIZZA I DATI PRIMA DI ONE-HOT ENCODING (CRITICO!)
    from sklearn.preprocessing import StandardScaler
    numeric_cols = [col for col in X_train.columns if col != 'country']
    scaler_data = StandardScaler()
    X_train[numeric_cols] = scaler_data.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler_data.transform(X_test[numeric_cols])
    
    X_train = pd.get_dummies(X_train, columns=['country'], drop_first=True, dtype=int)
    X_test = pd.get_dummies(X_test, columns=['country'], drop_first=True, dtype=int)
    

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MLP( [X_train.shape[1]] + [64]*3 + [1],nn.ReLU ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler()  # GradScaler per mixed precision
    

    
    train_dataset = TensorDataset(torch.tensor(X_train.to_numpy(), dtype=torch.float32).to(device),
                                  torch.tensor(y_train.to_numpy(), dtype=torch.float32).to(device))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    for epoch in range(700):
        model.train(True)  
        running_loss = 0.0

        # Ciclo sui batch di dati
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()  

            # Mixed Precision Training con autocast
            with torch.amp.autocast(device_type=device.type):
                outputs = model(inputs)  
                loss = torch.mean((outputs.squeeze() - labels) ** 2)
            
            # Backward con GradScaler + GRADIENT CLIPPING
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # Unscale prima del clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradienti
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:  # Stampa ogni 10 epoch
            print(f"Epoch {epoch+1}/700, Loss: {avg_loss:.4f}")
    
    
    # Store original country labels before one-hot encoding for filtering results
    X_test_with_country = X_test.copy()
    
    # Convert to tensors for prediction (all data at once)
    X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32).to(device)
    y_pred_all = model(X_test_tensor).detach().cpu().numpy().squeeze()
    
    # Get predictions per country
    # Need to retrieve country info from before one-hot encoding
    X_original_test = X[ X['year'] >= 2024 ]['country'].reset_index(drop=True)
    countries = X_original_test.unique()
    
    results = []
    for country in countries:
        print("COUNTRY: ",country)
        # Filter predictions for this country
        country_mask = (X_original_test == country).values
        y_pred = y_pred_all[country_mask]
        y_test_new = y_test.reset_index(drop=True)[country_mask]
        results.append(plot_metrics(y_test_new,y_pred,country,model_name="Neural Network"))
        print()
    return results




if __name__=="__main__":

    
    res_ensemble = ensemble()
    results_df = pd.DataFrame(res_ensemble)
    results_df.to_csv('results_ensemble.csv', index=False)
    
    results_nn = neural_network()
    results_df_nn = pd.DataFrame(results_nn)
    results_df_nn.to_csv('results_neural_network.csv', index=False)
    print(results_df)
    print(results_df_nn)