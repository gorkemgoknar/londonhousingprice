London House Price prediction Demo project

Note: this project is intended for pipeline generation for training and inference 

![Alt text](eda_images/london_eda3.png?raw=true "London House Prices")

##Building Docker images
### To build Trainer 
* Rename  Dockerfile_Trainer to Dockerfile
* Run in interactive environment
* docker run -it -v /yourworkdir:/tf/workdir -w /tf/workdir -p 8888:8888  IMAGEID bash
*
### To build default DNN model just run
python train.py  
If you need to fetch new data from GCP you need to provide credential file
python train.py --credential_file credential_file.json 


### To build Inference API
* Rename  Dockerfile_Inference to Dockerfile
* Run image which serves model 
* docker run -d -p 8080:8080 IMAGEID
* use localhost:8080 to access serving api
Api docs and sample can be found in localhost:8080/apidocs

### Basic EDA
#### Seasonality
![Alt text](eda_images/london_eda1.png?raw=true "Seasonality")
#### Area effect on price
![Alt text](eda_images/london_eda2.png?raw=true "Area effect on price")
