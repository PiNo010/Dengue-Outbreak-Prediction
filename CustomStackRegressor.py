import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 

N_FOLDS = 6
class StackRegressor():
    def __init__(self,models:list,meta_model,cat_features,):
        self.models = models
        self.meta_model = meta_model
        self.cat_features = cat_features
        self.trained_models = []
        
        
    def copy(self):
        return StackRegressor(self.models,self.meta_model,self.cat_features)
        
    def fit(self,X,y):
        kf = KFold(n_splits=N_FOLDS,shuffle=True)
        folds = []
        y_tot = []
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            predictions = []    
            print(f"FOLD {fold_idx+1}/{N_FOLDS}")
            X_tr, X_te = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[val_idx]
            # print(X_tr)
            for model in self.models:
                model_inside = model.copy()
                model_inside.fit(X_tr,y_tr,cat_features=self.cat_features)
                self.trained_models.append(model_inside)
    
                predictions.append(model_inside.predict(X_te))
            y_tot.extend(y_te)

            folds.append(np.column_stack(predictions))
            
        stack_folds = np.vstack(folds)
   
        self.meta_model.fit(stack_folds,y_tot)
        
        self.trained_models = []
        for base_model in self.models:
            m = base_model.copy()
            m.fit(X, y, cat_features=self.cat_features)
            self.trained_models.append(m)
        
        print("Training Complete")
        

    def predict(self,X):
        predictions = []
        for model in self.trained_models:
            predictions.append(model.predict(X))
            
        # print("SINGLE PREDS: ",predictions)
        meta_features = np.column_stack(predictions)
       
        return self.meta_model.predict(meta_features) 
    
   
            
            
        
            
            
        