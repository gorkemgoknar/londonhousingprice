import numpy as np
import tensorflow as tf
import pickle

class InferenceModel(object):
    """
    Class for model
    loads model on first init
    uses predict to predict via pipeline
    """
    def __init__(self):
        self.model = model_loader()
        self.pipe = make_sklearn_pipeline()
        self.encoder = encoder_loader()
        self.graph = tf.Graph()

    def parse_input(self,input):
        #maybe csv
        #maybe txt...
        return input

    def predict(self,input):
        x = self.parse_input(input)

        x_test = self.pipe.fit_transform(input)
        x_test = self.encoder.transform(x_test)
        x_test = np.asarray(x_test).astype(np.float32)
        return self.model.predict(x_test)

def model_loader():
    # It can be used to reconstruct the model identically.
    model = tf.keras.models.load_model("model/model.h5")
    print(model.summary)

    return model

def encoder_loader():
    with open('model/encoder.pickle', 'rb') as f:
        encoder = pickle.load(f)
        return encoder

def make_sklearn_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import ColumnTransformer
    from pipeline_functions import CopyData,DateFeatureEnhancer,AddressFeatureEnhancer,DropColumns

    rng = np.random.RandomState(0)

    drop_duplicate_cols = ["address", "date", "price"]
    cols_to_drop = ["longitude", "latitude", "address", "date", "locality_address","property_detail"]


    pipe = make_pipeline(
        CopyData(),
        DateFeatureEnhancer(col="date"),
        AddressFeatureEnhancer(col="address"),
        DropColumns(cols=cols_to_drop),

        ##experimental, instead of one hot encode use text based prediction.
        ##needs transformer model
        # ConvertToText()

    )

    return pipe
