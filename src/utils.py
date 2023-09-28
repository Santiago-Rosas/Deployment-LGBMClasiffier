from sklearn.pipeline import Pipeline
from sklearn. metrics import ConfusionMatrixDisplay, confusion_matrix
from joblib import dump
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt 

                    ###updatin the model and scaler 

def update_model(model) -> None: 
    dump(model, 'models/model.pk')

def update_scaler(scaler) -> None:  
    dump(scaler, 'models/scaler.pk')

def update_columns_equivalence(columns)->None:
     dump(columns, 'models/column_equivalence.pk')


                          ###function for report of metics 

def save_simple_metrics_report(train_score: float, test_score: float, validation_score: float, f1:float, model) -> None:
    with open('report.txt', 'w') as report_file:
        report_file.write('# Model Pipeline Description'+'\n')       
        report_file.write(f'### Molde LGBM with: {model.best_params_}'+'\n')
        report_file.write(f'### Train Score: {train_score}'+'\n')
        report_file.write(f'### Test Score: {test_score}'+'\n')
        report_file.write(f'### Accuracy Score: {validation_score}'+'\n')
        report_file.write(f'### F1_Score: {f1}'+'\n')

def get_confussion_matrix(y_real: pd.Series, y_pred: pd.Series) ->None:
    cm = confusion_matrix(y_real, y_pred) 
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(8)
    ax.set_title('Model prediction')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    fig.savefig('prediction_behavior.png')
    
 

                     #####DETECTING OUT_LAYERS function 


def outlier_thresholds(dataframe=pd.DataFrame, variable=str):
    quartile1 = dataframe[variable].quantile(0.01) ###in this case i will youse tyhsi values as a quartile becouse i dont want to loss to much info 
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit.round(), up_limit.round()

##suum all out layers per feature

##identifie the index of the out layer 
def index_out_layers(dataframe, variable=str):
    index=[]
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    index.append(list(np.where((dataframe[variable] < low_limit) | (dataframe[variable] > up_limit))[0]))
    return index[0]


###remove all out layers from a df 

def remove_out_layers_from_df(dataframe, variable=list)->pd.DataFrame:
    all_Out_layers=[]
    for i in variable:
        index=index_out_layers(dataframe, variable)
        for j in index:
            if j not in all_Out_layers:
                all_Out_layers.append(j)
        return dataframe.drop(dataframe.index[all_Out_layers])  


