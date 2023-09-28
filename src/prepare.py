from dvc import api
import pandas as pd 
from io import StringIO
import sys 
import logging
from pandas.core.tools import numeric

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

logging.info('Fetching data...')

smoking_data = api.read('dataset/smoking_drinking.csv', remote="dataset-storage")

df= pd.read_csv(StringIO(smoking_data))

df.to_csv('dataset/data.csv',index=False)

logger.info('Data Fetched and prepared...')