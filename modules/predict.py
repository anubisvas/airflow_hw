import json
import os
from datetime import datetime

import dill
import glob
import pandas as pd
import logging
path = os.environ.get('PROJECT_PATH', '/opt/airflow')
logging.info(glob.glob(f'{path}/data/models/*.pkl'))
model_filepath = f'{path}/data/models/cars_pipe.pkl'


def predict():
    with open(model_filepath, 'rb') as file:
        model = dill.load(file)
    preds_df = pd.DataFrame(columns=['car_id', 'preds'])
    for item in glob.glob(f'{path}/data/test/*.json'):
        with open(item, 'r') as fin:
            form = json.load(fin)
            df = pd.DataFrame.from_dict([form])
            y = model.predict(df)
            preds = pd.DataFrame({'car_id':df['id'],'preds':y})
            preds_df = pd.concat([preds_df,preds])
    preds_filename = f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    preds_df.to_csv(preds_filename, index=False)
    logging.info(f"predictions saved as {preds_filename}")
    return preds_df

if __name__ == '__main__':
    predict()


