London House Price prediction Demo project

To build Trainer 
* Rename  Dockerfile_Trainer to Dockerfile
* Run in interactive environment
* docker run -it -v /yourworkdir:/tf/workdir -w /tf/workdir -p 8888:8888  IMAGEID bash
To build default DN model just run
python train.py  
If you need to fetch new data from GCP you need to provide credential file
python train.py --credential_file credential_file.json 


To build Inference API
* Rename  Dockerfile_Inference to Dockerfile
* Run image which serves model 
* docker run -d -p 8080:8080 IMAGEID
* use localhost:8080 to access serving api
Api docs and sample can be found in localhost:8080/apidocs