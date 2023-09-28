# Deployment of LGBM model 

In this project we use smoking_drinking.csv data and LGBM Classifier for Drinker and not Drinker prediction

   
## WorkFlow
The steps for this project:

1. Notebook  with  EDA and all the steps for Feature selection, Sacling data, Model selection, and Hyperparameter tunning

2. Data Version Control for models and data with DVC

3. Creation of app with FastAPI and testing app with pytest

4. Containerization of the app with Docker

5. Workflow of Continuous Training, Continuous Integration and Continuous Deployment using Github Actions and CML  

6. Container Deployment with Cloud Run  

## For running app in localhost
1. Clone the repository: 
```bash
git@github.com:Santiago-Rosas/Deployment-LGBMClasiffier.git
```
2. Create the docker image: 
```bash
docker build . -t model_lgbm:v1
```
3. Build the docker container: 

```bash
docker run -d -p 8000:8000 model_lgbm:v1
```
