from utils import update_model, save_simple_metrics_report, get_confussion_matrix,remove_outliers_from_df, update_columns_equivalence,update_scaler
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, cross_validate
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import warnings
import logging
import sys
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

logger.info('Loading Data...')

df = pd.read_csv('dataset/data.csv')

                       ##preparing data

logger.info('Preparing Data...')

numeric_cols=[]
categorical_transformed=[]
categorical_untransformed=[]
for i,col in enumerate(df.columns):
    if df[col].dtype =='object':
        categorical_untransformed.append(col)
    if df[col].dtype =='float64' or df[col].dtype == "int": 
        if len(df[col].unique()) > 10:
            numeric_cols.append(col) 
        else:
            categorical_transformed.append(col)

df_2=remove_outliers_from_df(df,numeric_cols)
df_2.drop(columns="LDL_chole",inplace=True)


                          ####trasmforming data (categorical variabels) 

logger.info('Transforming Data...')

column_equivalence = {}
for i, column in enumerate(df_2.dtypes):
    if column == "object":
        categorical_column = df_2[df_2.columns[i]].astype("category")
        current_column_equivalence = dict(enumerate(categorical_column.cat.categories))
        column_equivalence[i] = dict((v,k) for k,v in current_column_equivalence.items())
        df_2[df_2.columns[i]] = categorical_column.cat.codes



                          ###Splitting and subsampling 


logger.info('Spliting  and undersampling Data...')

X= df_2.drop(columns="DRK_YN")
y=df_2["DRK_YN"]

undersample = RandomUnderSampler(random_state=42,sampling_strategy={0:20000,1:20000})
X_over , y_over = undersample.fit_resample(X,y)

X_train, X_test, y_train, y_test = train_test_split( X_over, y_over, test_size=0.33, random_state=42)


                             ####scaling data 


logger.info('Scaling')

columns_scale=[]
not_scale=[]
for index, column in enumerate(X.columns):
    if len(df_2[column].unique()) > 5 and X[column].max() > 2:
        columns_scale.append(index)
    else:
        not_scale.append(index)

scaler = ColumnTransformer(
[("scale",StandardScaler(),columns_scale)],remainder= 'passthrough'
)

scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

                             

                    #####Model and hyperparameters 

logger.info('model hyerparameter tunning..')

param={
          "max_depth": list(range(3,10)),
          "learning_rate":[0.001, 0.01,0.1],
          "num_iterations":[300],
          "num_leaves":[10,15,31,40,50,60,100],
          'verbose':[-1]
          }

grid_reg = GridSearchCV(LGBMClassifier(), param,scoring='accuracy', cv=4)

grid_reg.fit(X_train, y_train.values.ravel())


model=grid_reg.best_estimator_
print(model.get_params)

                        ###Cross validation 


logger.info('Cross validating with best model...')
final_result = cross_validate(model, X_train, y_train, return_train_score=True, cv=5)


train_score = np.mean(final_result['train_score'])
test_score = np.mean(final_result['test_score'])


logger.info(f'Train Score: {train_score}')
logger.info(f'Test Score: {test_score}')



                   ####updating the model 

logger.info('Updating model, sacler and column_equivalence...')
update_model(model)
update_scaler(scaler)
update_columns_equivalence(column_equivalence)


            #######evaluating the momdel 


logger.info('Generating model report...')
validation_score = model.score(X_test, y_test)
f1=f1_score(y_test,model.predict(X_test))
save_simple_metrics_report(train_score, test_score, validation_score,f1, grid_reg)

y_test_pred = model.predict(X_test)
get_confussion_matrix(y_test, y_test_pred)

logger.info('Training Finished')

