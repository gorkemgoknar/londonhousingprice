#! /usr/bin/env python3
# -*- coding: UTF-8 -*-

import pickle
from flask import Flask, request, render_template

#from flask_restx import Resource, Api

from flasgger import Swagger

import numpy as np
import pandas as pd

from model.model_loader import InferenceModel

ml_api = Flask(__name__)
swagger = Swagger(ml_api)

global model
model = InferenceModel()


@ml_api.route('/')
def index_view():
    return render_template('index.html')

@ml_api.route("/predict", methods=["GET","POST"])
def predict():
    """
    Endpoint predict House prices in London

    Sample
    address	:Flat 29, Mulberry Court, 1, School Mews, London,...
    type : Flat
    bedrooms: 0
	latitude: 51.51061
	longitude: -0.05992
	area: E1
	price: 500000
	tenure: Leasehold
	is_newbuild: 1
	date: 2011-03-02 00:00:00+00:00
    ---
    parameters:
        - name: address
          in: query
          required: true
          schema:
              type: string
              #enum: [approved, pending, closed, new]
              example: Flat 29, Mulberry Court, 1, School Mews, London, Greater London E1 0EW
        - name: type
          in: query
          required: true
          schema:
              type: string
              enum: [flat, Terraced]
              example: flat
        - name: bedrooms
          in: query
          required: true
          schema:
              type: int
              enum: [0, 1,2,3,4]
              example: 1
        - name: area
          in: query
          required: true
          schema:
              type: string
              enum: [E1, SE15]
              example: E1
        - name: tenure
          in: query
          type: string
          required: true
          schema:
              type: string
              enum: [Leasehold, Freehold]
              example: Leasehold
        - name: is_newbuild
          in: query
          required: true
          schema:
              type: int
              enum: [0, 1]
              example: 1
        - name: date
          in: query
          type: string
          required: true
          schema:
              type: string
              enum: [1999-01-05, 2020-01-05]
              example: 2016-01-05
    responses:
        200:
          description: Number indicating  price prediction

    """

    address = request.args.get("address")
    type = request.args.get("type")
    bedrooms = request.args.get("bedrooms")
    area = request.args.get("area")
    tenure = request.args.get("tenure")
    is_newbuild = request.args.get("is_newbuild")
    #assert 1/0
    date = request.args.get("date")
    print(date)
    ##assert date

    latitude = request.args.get("latitude")
    longitude = request.args.get("longitude")

    if latitude is None or longitude is None:
        latitude = 51.51061
        longitude = -0.05992

    input_df = pd.DataFrame({
        'address': [address],
        'type': [type],
        'bedrooms': [bedrooms],
        'area': [area],
        'tenure': [tenure],
        'is_newbuild': [is_newbuild],
        'latitude': [latitude],
        'longitude': [longitude],
        'date': [date],

    })

    print(input_df)

    #input_data = np.array( [[address,type, bedrooms,area,tenure,is_newbuild,date,latitude,longitude]])
    out = model.predict(input_df)
    print(out[0][0])
    return str(out[0][0])

if __name__ == "__main__":
    ml_api.run(host="0.0.0.0", port=8888)