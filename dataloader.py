import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

import os
import shutil
from pathlib import Path

from sqlalchemy.engine import create_engine
#from config import bigquery_uri, gcp_credentials_path
#from config import bigquery_dataset, bigquery_table
import pprint


def load_data(credential_file=None,gcp_project=None,data_folder="data_parquet"):
    if credential_file is not None and  gcp_project is not None:
        #get from gcp
        return load_data_from_bigquery(credential_file,gcp_project)
    else:
        return load_data_from_folder(data_folder)

def load_data_from_folder(data_folder="data_parquet"):
    try:
        data = pd.read_parquet(data_folder)
        logging.info(f"Loaded dataset by shape {data.shape} from {data_folder}")
    except:
        raise FileNotFoundError("Cannot load parquet file from data_folder " )


    return data

def load_data_from_bigquery(credential_file,gcp_project):
    bigquery_uri = f'bigquery://{gcp_project}/london_house_prices'

    logging.info(f"Loading from bigquery via {credential_file}")
    #load credentials
    bigquery_table = "london_house_prices"
    # set sqlalchemy engine
    engine = create_engine(
        bigquery_uri,
        credentials_path=credential_file
    )

    # load chunk reader for pandas
    london_df_chunks = pd.read_sql_table(
        bigquery_table,
        con=engine,
        chunksize=50000
    )


    #save to default folder path data_parquet
    parquet_folder= 'data_parquet'

    ##cleanup

    logging.info("Clearing folders for saving data..")
    parquet_path= os.path.join(parquet_folder)
    Path("model").mkdir(parents=True, exist_ok=True)

    if Path(parquet_path).is_dir():
        shutil.rmtree(parquet_path)
    Path(parquet_path).mkdir(parents=True, exist_ok=True)
    #os.mkdir(parquet_path)

    csv_path = os.path.join("data_csv")
    if Path(csv_path).is_dir():
        shutil.rmtree(csv_path)
    Path(csv_path).mkdir(parents=True, exist_ok=True)

    #TODO: clean up folder before saving
    logging.info("Saving data in chunks..")
    ##to use with chunk
    count = 0
    for chunk in london_df_chunks:
        logging.debug(f"Loading chunk {count}")
        #csv is also saved to data_csv folder
        chunk.to_csv(os.path.join("data_csv",bigquery_table + ".csv"), mode='a', sep=',', encoding='utf-8')

        file_path = parquet_path + '/part.%s.parquet' % (count)
        chunk.to_parquet(file_path, engine='pyarrow')
        count += 1

    logging.info(f"Wrote files to folders parquet:{parquet_path} and csv:data_csv ")

    ##now read parquet and return
    return load_data_from_folder(parquet_folder)