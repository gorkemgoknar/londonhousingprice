import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import mean_absolute_error
from joblib import Memory
#from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from tempfile import mkdtemp
import datetime
from dateutil.parser import parse
import inspect
from numbers import Number
import math

import zlib
import zipfile


class CopyData(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return data.copy()


class DropDuplicates(BaseEstimator, TransformerMixin):
    """
    Drop duplicate rows
    """

    def __init__(self, cols=[]):
        # if no columns given will do nothing
        self.column_names_to_drop = cols

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        # given column names to check for duplication return cleaned dataframe
        return data.drop_duplicates(subset=self.column_names_to_drop, keep='first')


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols=[]):
        self.cols = cols

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return data.drop(self.cols, axis=1)


class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    Remove values below minimum and above maximum quantiles.
    #for our scenarioy 0.01 and 0.99 loses data, we need to remove extreme outliers only

    """

    def __init__(self, col="column_name", min_quantile=0.01, max_quantile=0.99):
        self.min_quantile = min_quantile
        self.max_quantile = max_quantile
        self.column_name = col
        self.min_value = 0
        self.max_value = 0

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        # check if we are required to use standart deviation
        self.min_value, self.max_value = data[self.column_name].quantile([self.min_quantile, self.max_quantile])

        return data.query(
            "{column_name} > {min_val} and {column_name} < {max_val}".format(
                column_name=self.column_name,
                min_val=self.min_value,
                max_val=self.max_value))


# Feature engineering

class DateFeatureEnhancer(BaseEstimator, TransformerMixin):
    def __init__(self, col="date"):
        print("DateFeatureEnhancer")
        self.base_date = pd.to_datetime('1800-01-01T00:00:00.000Z')
        self.date_col_name = col

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        data[self.date_col_name] = pd.to_datetime(data[self.date_col_name])
        # generate date features

        data["yearSold"] = data["date"].dt.year
        # generate decade info to get rid of year and cover at least 2020-2029
        # data["decadeSold"] = data["yearSold"]- (data["yearSold"] %10)
        # data = data.drop("yearSold",axis=1)

        data["monthSold"] = data["date"].dt.month

        # exact day should not be important
        ##data["daySold"]= data["date"].dt.day # may not need

        data["weekSold"] = data["date"].dt.isocalendar().week
        data["quarterSold"] = data["date"].dt.quarter

        # day of week is relevant
        data["dayOfWeekSold"] = data["date"].dt.dayofweek

        # capture day from basedate information
        data["days_from_basedate"] = data[self.date_col_name].apply(lambda x: (x - self.base_date).days)

        return data


import re


def split_on_firstnumber(s):
    splitted = re.split(r'(\d+)', s)
    return (splitted[0], splitted[1])


class AddressFeatureEnhancer(BaseEstimator, TransformerMixin):

    def get_address_details(self, add_list):
        """
        Flat B	 Property subdivision
        39 Acacia Avenue	 Property number and street address
        North End	 Locality address
        Silhurst	 Post town
        Loamshire	 County
        SH15 6BP	 Unit postcode

        ##capture regex: (.+)?,(.+)?,(.+)((?<=, )?.*)
        ##Group 1(Flat 1610, Defoe House, 123, City Island Way), Group 2 (London), Group 3(Greater London E14 0TW)
        #Group 1 may not have house name/property subdivision but street name may be a feature

        """
        # last one is county + unit postcode
        # e.g Greater London E14 0TW
        postcode = add_list[-1].strip()

        # e.g. London
        post_town = add_list[-2].strip()

        # will have street name/avenue name
        locality_address = add_list[-3].strip()

        # property number only or property name and number info
        property_detail = add_list[:-3]

        ##number -> get from property_detail until 1st number

        return property_detail, locality_address, post_town, postcode

    def __init__(self, col="date"):
        print("Adress Feature Enhancer")
        self.address_col_name = col

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        data["address_splitted"] = data["address"].str.split(',')
        data["property_detail"], data["locality_address"], data["post_town"], data["postcode"] = zip(
            *data['address_splitted'].map(self.get_address_details))

        data = data.drop("address_splitted", axis=1)

        # from postcode information extract city info
        data["metropolitan_area"] = data["postcode"].apply(lambda x: " ".join(x.split(" ")[0:-2]))

        # area info is same as "area" column...
        data["area_from_address"] = data["postcode"].apply(lambda x: x.split(" ")[-2:])

        # get street information
        data["street_post"] = data["area_from_address"].apply(lambda x: x[1][0])

        # area + street_post is actuall street info which is important!!
        # and should be similar to street name
        # may as well join this info and tokenize it!

        ##especially for expensive houses
        # data["area_first_part"], data["area_second_part"] = zip(*data['area'].map(split_on_firstnumber))

        # done with postcode, area info already exit
        data = data.drop("postcode", axis=1)
        data = data.drop("area_from_address", axis=1)

        # interesting maybe only get its last word
        ##data["locality_ending"] = data["locality_address"].apply(lambda x: x.split(" ")[-1])
        # data = data.drop("locality_address",axis=1)

        """
        ##MAYBE use this only in fit?
        ##and check on transform?
        ##street name directly affects price, top 20 are most expensive!
        #using this introduced data leakage, cannot do this on inference
        #we will use postcode information instead of this
        sum_by_locality_df= (data[["price","locality_address"]].groupby("locality_address").median().reset_index()
                            .sort_values(by = ['price'], ascending=[False])
                            ).head(20)

        #likely Road, Avenue etc..
        #get top 10
        best_localities=  sum_by_locality_df["locality_address"].head(20).tolist()

        for list_member in best_localities:
            col_name= "locality_address"+"_"+list_member
            data[col_name] = 0
            data.loc[data["locality_address"]== list_member, col_name] = 1

        data = data.drop("locality_address",axis=1)
        """
        print(data.columns)

        return data

class LabelEncodeObjects(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data):
        for c in data.columns:
            if data[c].dtype == 'object':
                lbl = LabelEncoder()
                lbl.fit(list(data[c].values))
                data[c] = lbl.transform(list(data[c].values))
        return data


class LabelEncodeCols(BaseEstimator, TransformerMixin):
    def __init__(self, cols=[]):
        self.cols = cols

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        for col in self.cols:
            lbl = LabelEncoder()
            lbl.fit(list(data[col].values))
            data[col] = lbl.transform(list(data[col].values))
        return data


class OneHotEncoderDummy(BaseEstimator, TransformerMixin):
    def __init__(self, cols=[]):
        print("One Hot encoder")
        print(cols)
        self.cols = cols

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        for col in self.cols:
            dummies = pd.get_dummies(data[col], prefix=col)
            data = pd.concat([data, dummies], axis=1)
            data = data.drop(col, axis=1)

        return data


class ConvertToFloat(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return np.asarray(data).astype(np.float32)


class ConvertToText(BaseEstimator, TransformerMixin):
    def __init__(self, cols=[]):
        self.cols = cols

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        df2 = pd.DataFrame(index=data.index)
        df2["text"] = ""
        for index, row in data.iterrows():
            out = "Price for "
            for k, v in row.items():
                if k == "address":
                    continue
                if k == "price":
                    continue
                if k == "date":
                    continue
                else:
                    out += "" + k + ":" + str(v) + " "
            out += " on date: " + str(row["date"])
            df2.at[index, "text"] = out

        return df2

