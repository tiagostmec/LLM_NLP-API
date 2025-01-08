import os
import uvicorn
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI

from routers.routes_anomalies import router as router_anomalies

from version import VERSION
from api_configs import PORT


load_dotenv(find_dotenv())
ENVIRONMENT_NAME = os.getenv("ENVIRONMENT_NAME", None)

if (ENVIRONMENT_NAME is not None) and (ENVIRONMENT_NAME.startswith("production")):
    # Disable swagger and redocs on production enviroment.
    app = FastAPI(docs_url=None, redoc_url=None)
else:
    app = FastAPI(title="AOI - EOT API", description="API para inspeção cosmética.", version=VERSION)

# Insert the routers on the app 
app.include_router(router_anomalies)


if __name__ == "__main__":
    API_PORT = os.getenv("API_PORT", PORT)
    uvicorn.run(app=app, port=int(API_PORT))
