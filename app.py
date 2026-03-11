import sys
import os
import re
import socket
from urllib.parse import urlparse

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGODB_URL_KEY")
print(mongo_db_url)
import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile,Request, HTTPException, Form
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from networksecurity.utils.main_utils.utils import load_object

from networksecurity.utils.ml_utils.model.estimator import NetworkModel


client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

from networksecurity.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from networksecurity.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/ui")


def _load_predictor():
    preprocessor = load_object("final_model/preprocessor.pkl")
    model = load_object("final_model/model.pkl")
    expected_columns = list(getattr(preprocessor, "feature_names_in_", []))
    network_model = NetworkModel(preprocessor=preprocessor, model=model)
    return network_model, expected_columns


def _normalize_input_dataframe(df: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    if "Result" in df.columns:
        df = df.drop(columns=["Result"])

    if expected_columns:
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_columns}"
            )
        df = df[expected_columns]

    return df


def _is_ip(hostname: str) -> bool:
    if not hostname:
        return False
    try:
        socket.inet_aton(hostname)
        return True
    except OSError:
        return False


def _extract_url_features(raw_url: str, expected_columns: list[str]) -> pd.DataFrame:
    url = raw_url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="Please enter a URL.")

    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = f"http://{url}"

    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    scheme = (parsed.scheme or "").lower()

    shorteners = {
        "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "is.gd",
        "buff.ly", "adf.ly", "bit.do", "cutt.ly", "rebrand.ly"
    }
    explicit_port = parsed.port
    non_standard_port = explicit_port is not None and not (
        (scheme == "http" and explicit_port == 80) or
        (scheme == "https" and explicit_port == 443)
    )

    if hostname:
        host_parts = hostname.split(".")
        subdomain_count = max(0, len(host_parts) - 2)
    else:
        subdomain_count = 0

    if subdomain_count <= 1:
        having_sub_domain = 1
    elif subdomain_count == 2:
        having_sub_domain = 0
    else:
        having_sub_domain = -1

    url_length = len(url)
    if url_length < 54:
        url_length_signal = 1
    elif url_length <= 75:
        url_length_signal = 0
    else:
        url_length_signal = -1

    short_service = -1 if any(hostname.endswith(s) for s in shorteners) else 1
    at_symbol = -1 if "@" in url else 1
    double_slash_redirecting = -1 if "//" in (parsed.path or "") else 1
    prefix_suffix = -1 if "-" in hostname else 1
    ssl_state = 1 if scheme == "https" else -1
    https_token = -1 if ("https" in hostname and scheme != "https") else 1
    abnormal_url = 1 if hostname else -1

    feature_map = {
        "having_IP_Address": -1 if _is_ip(hostname) else 1,
        "URL_Length": url_length_signal,
        "Shortining_Service": short_service,
        "having_At_Symbol": at_symbol,
        "double_slash_redirecting": double_slash_redirecting,
        "Prefix_Suffix": prefix_suffix,
        "having_Sub_Domain": having_sub_domain,
        "SSLfinal_State": ssl_state,
        "Domain_registeration_length": 0,
        "Favicon": 1,
        "port": -1 if non_standard_port else 1,
        "HTTPS_token": https_token,
        "Request_URL": 0,
        "URL_of_Anchor": 0,
        "Links_in_tags": 0,
        "SFH": 0,
        "Submitting_to_email": -1 if "mailto:" in url.lower() else 1,
        "Abnormal_URL": abnormal_url,
        "Redirect": 0,
        "on_mouseover": 1,
        "RightClick": 1,
        "popUpWidnow": 1,
        "Iframe": 1,
        "age_of_domain": 0,
        "DNSRecord": 1 if hostname else -1,
        "web_traffic": 0,
        "Page_Rank": 0,
        "Google_Index": 0,
        "Links_pointing_to_page": 0,
        "Statistical_report": 1,
    }

    if expected_columns:
        missing_features = [col for col in expected_columns if col not in feature_map]
        if missing_features:
            raise HTTPException(
                status_code=500,
                detail=f"Feature mapping missing for: {missing_features}"
            )
        ordered = {col: feature_map[col] for col in expected_columns}
    else:
        ordered = feature_map

    return pd.DataFrame([ordered])


@app.get("/ui")
async def ui_page(request: Request):
    return templates.TemplateResponse("ui.html", {"request": request, "result": None, "error": None})

@app.get("/train")
async def train_route():
    try:
        train_pipeline=TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
@app.post("/predict")
async def predict_route(request: Request,file: UploadFile = File(...)):
    try:
        df=pd.read_csv(file.file)
        network_model, expected_columns = _load_predictor()
        df = _normalize_input_dataframe(df, expected_columns)
        print(df.iloc[0])
        y_pred = network_model.predict(df)
        print(y_pred)
        df['predicted_column'] = y_pred
        print(df['predicted_column'])
        #df['predicted_column'].replace(-1, 0)
        #return df.to_json()
        df.to_csv('prediction_output/output.csv', index=False)
        table_html = df.to_html(classes='table table-striped')
        #print(table_html)
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
        
    except HTTPException:
            raise
    except Exception as e:
            raise NetworkSecurityException(e,sys)


@app.post("/predict-url")
async def predict_from_url(request: Request, url: str = Form(...)):
    try:
        network_model, expected_columns = _load_predictor()
        df = _extract_url_features(url, expected_columns)
        y_pred = network_model.predict(df)
        prediction_value = int(y_pred[0])
        label = "Legitimate URL" if prediction_value == 1 else "Phishing URL"
        table_html = df.assign(predicted_column=y_pred).to_html(classes='table table-striped', index=False)
        return templates.TemplateResponse(
            "ui.html",
            {
                "request": request,
                "result": {
                    "url": url,
                    "prediction": prediction_value,
                    "label": label,
                    "table_html": table_html,
                },
                "error": None,
            },
        )
    except HTTPException as e:
        return templates.TemplateResponse("ui.html", {"request": request, "result": None, "error": e.detail})
    except Exception as e:
        return templates.TemplateResponse("ui.html", {"request": request, "result": None, "error": str(e)})

    
if __name__=="__main__":
    app_run(app,host="0.0.0.0",port=8000)
