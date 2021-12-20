"""
Train a DeepLearning model on London Housing dataset
Usage:
    $ python path/to/train.py --data dataset_parquet_folder
"""
import argparse
import logging
logger = logging.getLogger(__name__)


from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import mean_absolute_error
from joblib import Memory
#from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

from pipeline_functions import *


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Bidirectional,Embedding, LSTM
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


from model_loader import make_sklearn_pipeline

def clean_dataset(df, target_column):
    count_of_prev_rows = df.shape[0]

    drop_duplicate_cols = ["address", "date", "price"]

    reduced = DropDuplicates(cols=drop_duplicate_cols).transform(df)
    reduced = OutlierRemover(col=target_column, min_quantile=0.001, max_quantile=0.999).transform(reduced)

    count_of_final_rows = reduced.shape[0]
    count_of_duplicate_rows = count_of_prev_rows - count_of_final_rows

    logging.info(f"{count_of_final_rows} rows saved, {count_of_duplicate_rows} duplicates removed, {100 - 100 * count_of_duplicate_rows / count_of_prev_rows :.2f}% reduction")
    return reduced


def create_model(layers=[256,128,64]):
    logging.info("Creating model")
    # can as well make neuron size according to available features (X[1])
    model = Sequential()
    for i,neuron_size in enumerate(layers):
        model.add(Dense(neuron_size, activation="relu"))
        model.add(Dropout(0.1))

    model.add(Dense(20, activation="relu"))
    model.add(Dense(1, activation="linear"))


    return model

def train_model(args):

    ##LOAD DATASET
    #TODO: add file load or force GCP refresh option
    import dataloader
    logging.info("Getting Dataset")
    if not args.reload_data:
        gcp_credential_file = None
    else:
        gcp_credential_file = args.credential_file
    data = dataloader.load_data(credential_file=gcp_credential_file, gcp_project=args.gcp_project, data_folder=args.datapath)

    if data is None :
        raise FileNotFoundError('Data file cannot be loaded')

    ##CLEAN DATA
    logging.info("Cleaning dataset")
    #clean dataset and remove outliers
    dataset = clean_dataset(data, "price")
    del data

    logging.info("Splitting train test")
    target_column = "price"
    train_cols = [x for x in dataset.columns if x != target_column]

    X_train, X_test, y_train, y_test = train_test_split(dataset[train_cols], dataset[target_column], test_size=0.01,
                                                        random_state=42)

    logging.info(f"{X_train.shape[0]} rows in train, {X_test.shape[0]} rows in test")


    pipe = make_sklearn_pipeline()

    x_train = pipe.fit_transform(X_train)  # y_train

    # SAVE ENCODER to use in test
    # OneHotEncoder(sparse=False,handle_unknown = "ignore"),
    # better to use ColumnTransformer as can choose columns
    # ColumnTransformer( [("OneHotEncode", OneHotEncoder(sparse=False,handle_unknown = "ignore"),cols_to_onehot_encode)] , remainder="passthrough" ),
    cols_to_onehot_encode = ["area", "street_post", "bedrooms", "tenure", "metropolitan_area", "post_town", "type"]

    encoder = ColumnTransformer(
        [
            ("OneHotEncode", OneHotEncoder(sparse=False, handle_unknown="ignore"), cols_to_onehot_encode)
        ]
        , remainder="passthrough")
    x_train = encoder.fit_transform(x_train)
    x_train = np.asarray(x_train).astype(np.float32)
    import pickle
    with open('model/encoder.pickle', 'wb') as f:
        pickle.dump(encoder, f)

    print("TEST")
    x_test = pipe.fit_transform(X_test)  # y_Test

    x_test = encoder.transform(x_test)

    x_test = np.asarray(x_test).astype(np.float32)

    ##get model

    model = create_model(layers=[300,300,75])

    learning_rate = 0.005
    batch_size = args.batch_size
    epochs = args.epochs

    # loss functions
    # mse = tf.keras.losses.MeanSquaredError()
    # cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
    # msle = tf.keras.losses.MeanSquaredLogarithmicError()
    # huber loss, sensistive to outlier
    huber_loss = tf.keras.losses.Huber()

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    model.compile(optimizer, loss=huber_loss)

    from tqdm.keras import TqdmCallback

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.10,
        batch_size=batch_size,
        verbose=0, epochs=epochs, callbacks=[early_stop, TqdmCallback(verbose=0)])

    logging.info(f"Final validation loss: { str(history.history['val_loss']) } ")

    #test predict


    #save model
    from pathlib import Path
    Path("model").mkdir(parents=True, exist_ok=True)


    model_path = "model/model.h5"
    model.save(model_path)
    logging.info("Done saving model")

    #save history

    with open("model/history.pickle", 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    logging.info("Done saving history")


def main(args):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename='log_trainer.log',
                        filemode='w')

    train_model(args)


def parse_opt():
    parser = argparse.ArgumentParser(description='Training args for London House prediction')
    parser.add_argument('--datapath', type=str, default="data_parquet", help='location for parquet folder')
    parser.add_argument('--batch_size', type=int, default=256, help='Batchsize for training')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in model')
    parser.add_argument('--model_type', type=str, default="dense", help='Model type, dense or transformer')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--credential_file', type=str, default="config/gcp_credentials.json", help='GCP Credential json file')
    parser.add_argument('--gcp_project', type=str, default="mystical-accord-330011", help='GCP Project Name')
    parser.add_argument('--reload_data', type=int, default=0, help='If set will delete data folder and reload data from GCP')

    args =  parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_opt()
    main(args)
