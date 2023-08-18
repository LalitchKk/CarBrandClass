import pickle
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# import method predict_brand from code.py 
from app.code import predict_brand



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],
    allow_headers=["*"],
)
# use docker
model = pickle.load(open(f'model\model_XGB.pkl','rb'))

# use docker
end_hog = 'http://172.17.0.2:80/api/gethog'

@app.get("/")
def root():
    return {"message": "This is my api"}

@app.post("/api/carbrand")
async def read_str(request:Request):
    item = await request.json()
    hog = requests.get(end_hog,json=item)
    res = predict_brand(model,hog.json()['HOG'])
    return res
