####################################### IMPORT #################################
import json
import pandas as pd
import numpy as np
from PIL import Image
from loguru import logger
import sys
from io import BytesIO

from fastapi import FastAPI, File, status, Depends, Query
from fastapi.responses import RedirectResponse
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException

from database import engine, SessionLocal
import models
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import get_image_from_bytes
from app import detect_sample_model
from app import add_bboxs_on_img
from app import get_bytes_from_image

from app import get_positions
from app import get_path_dict
from app import get_row_pred

from ultralytics import YOLO

####################################### logger #################################

logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level=10,
)
logger.add("log.log", rotation="1 MB", level="DEBUG", compression="zip")

###################### FastAPI Setup #############################

# title
app = FastAPI(
    title="KOA-api",
    description="""Obtain colonies out of images and return image and json result""",
    version="2023.9.22",
)

# This function is needed if you want to allow client requests 
# from specific domains (specified in the origins argument) 
# to access resources from the FastAPI server, 
# and the client and server are hosted on different domains.
origins = [
    "http://localhost",
    "http://localhost:8008",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

## Add database
models.Base.metadata.create_all(bind=engine)

def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()    

@app.on_event("startup")
def save_openapi_json():
    '''This function is used to save the OpenAPI documentation 
    data of the FastAPI application to a JSON file. 
    The purpose of saving the OpenAPI documentation data is to have 
    a permanent and offline record of the API specification, 
    which can be used for documentation purposes or 
    to generate client libraries. It is not necessarily needed, 
    but can be helpful in certain scenarios.'''
    openapi_data = app.openapi()
    # Change "openapi.json" to desired filename
    with open("openapi.json", "w") as file:
        json.dump(openapi_data, file)

# redirect
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


@app.get('/healthcheck', status_code=status.HTTP_200_OK)
def perform_healthcheck():
    '''
    It basically sends a GET request to the route & hopes to get a "200"
    response code. Failing to return a 200 response code just enables
    the GitHub Actions to rollback to the last version the project was
    found in a "working condition". It acts as a last line of defense in
    case something goes south.
    Additionally, it also returns a JSON response in the form of:
    {
        'healtcheck': 'Everything OK!'
    }
    '''
    return {'healthcheck': 'Everything OK!'}


######################### Support Func #################################

def crop_image_by_predict(image: Image, predict: pd.DataFrame(), crop_class_name: str,) -> Image:
    """Crop an image based on the detection of a certain object in the image.
    
    Args:
        image: Image to be cropped.
        predict (pd.DataFrame): Dataframe containing the prediction results of object detection model.
        crop_class_name (str, optional): The name of the object class to crop the image by. if not provided, function returns the first object found in the image.
    
    Returns:
        Image: Cropped image or None
    """
    crop_predicts = predict[(predict['name'] == crop_class_name)]

    if crop_predicts.empty:
        raise HTTPException(status_code=400, detail=f"{crop_class_name} not found in photo")

    # if there are several detections, choose the one with more confidence
    if len(crop_predicts) > 1:
        crop_predicts = crop_predicts.sort_values(by=['confidence'], ascending=False)

    crop_bbox = crop_predicts[['xmin', 'ymin', 'xmax','ymax']].iloc[0].values
    # crop
    img_crop = image.crop(crop_bbox)
    return(img_crop)


######################### MAIN Func #################################


@app.post("/img_object_detection_to_json")
def img_object_detection_to_json(file: bytes = File(...)):
    """
    Object Detection from an image.

    Args:
        file (bytes): The image file in bytes format.
    Returns:
        dict: JSON format containing the Objects Detections.
    """
    # Step 1: Initialize the result dictionary with None values
    result={'detect_objects': None}

    # Step 2: Convert the image file to an image object
    input_image = get_image_from_bytes(file)

    # Step 3: Predict from model
    predict = detect_sample_model(input_image)

    # Step 4: Select detect obj return info
    # here you can choose what data to send to the result
    detect_res = predict[['name', 'confidence']]
    objects = detect_res['name'].values

    result['detect_objects_names'] = ', '.join(objects)
    result['detect_objects'] = json.loads(detect_res.to_json(orient='records'))

    # Step 5: Logs and return
    logger.info("results: {}", result)
    return result

@app.post("/img_object_detection_to_img")
def img_object_detection_to_img(file: bytes = File(...)):
    """
    Object Detection from an image plot bbox on image

    Args:
        file (bytes): The image file in bytes format.
    Returns:
        Image: Image in bytes with bbox annotations.
    """
    # get image from bytes
    input_image = get_image_from_bytes(file)

    # model predict
    predict = detect_sample_model(input_image)

    # add bbox on image
    final_image = add_bboxs_on_img(image = input_image, predict = predict)

    # return image in bytes format
    return StreamingResponse(content=get_bytes_from_image(final_image), media_type="image/jpeg")

@app.post('/aquagar_predict_mariadb')
async def aquagar_predict_mariadb(timestamp: str, 
                 serial_num: str,
                 file: bytes = File(...)):
    
    # ATENTION: this step will need some coding. The plate_id is supposed to be read from the picture!
    plate_id = 'XXXX' 
    
    # Get image from bytes
    input_image = get_image_from_bytes(file)

    modelAgarsWells = YOLO('models/sample_model/model_agars_wells.pt')

    resultsAgars = modelAgarsWells(input_image)[0]
    allBoxes = resultsAgars.boxes.xyxy.numpy().astype(int)

    agarsPositions = get_positions(allBoxes[:, 0:2], 2, 3)
    # Sort the array first by the last column (index 3), then by the one before column (index 2)
    sorted_indices = np.lexsort((agarsPositions[:, 3], agarsPositions[:, 2]))
    # Here sorting is top-down, and left-right, so like:
    # 1 2 3
    # 4 5 6 
    allBoxesSorted = allBoxes[sorted_indices]

    # Perform pathogen prediction on all the agars
    modelColonies = YOLO('models/sample_model/model_all_augment.pt')
    pred_on_all_agars = []
    for box in allBoxesSorted:
        agarCrop = input_image.crop((box[0], box[1], box[2], box[3]))
        #agarCrop = input_image[box[1]:box[3], box[0]:box[2]]
        results = modelColonies.predict(agarCrop, conf = .6)
         # Colonies prediction on agarCrop
        path_dict = get_path_dict(results)        # Get count of each pathogen type
        pred_on_all_agars.append(path_dict)       # Append global list

    if len(pred_on_all_agars)==6:
        # The official column order for agars is: TCBS - MSA - BA
        upperRowTCBS, upperRowMSA, upperRowBA, lowerRowTCBS, lowerRowMSA, lowerRowBA= pred_on_all_agars[0], pred_on_all_agars[1], pred_on_all_agars[2], pred_on_all_agars[3], pred_on_all_agars[4], pred_on_all_agars[5]

        # After **some logic** we get to a final prediction
        upperRowPred, lowerRowPred = upperRowBA, lowerRowBA
    
    else:
        upperRowTCBS, upperRowMSA, upperRowBA, lowerRowTCBS, lowerRowMSA, lowerRowBA, upperRowPred, lowerRowPred = {}, {}, {}, {}, {}, {}, {}, {} 
    
    import mysql.connector
    import json

    # Replace with your MySQL connection details
    host =  '10.8.0.1'
    username = 'pere'
    password = 'Nemomola5'
    database_name =  'KOAPredictions'

    # Create a connection to the MySQL server
    db_connection = mysql.connector.connect(
        host=host,
        user=username,
        password=password,
        database=database_name
    )

    # Create a cursor to execute SQL commands
    cursor = db_connection.cursor()

    #test_pred_sample = {'assalmonicida': 0, 'pddamselae': 0, 'pdpiscicida': 0, 'sinniae': 0}
    #test_pred_sample = json.dumps(test_pred_sample)

    try: 
        # Common query for inserting predictions into MariaDB
        ''' 
        query = """
                INSERT INTO aquagar (id_maquina, PLATE_ID, TIME_STAMP, ROW,PRED_TCBS, PRED_MSA, PRED_BA, PRED, SERIAL_NUM)
                SELECT
                    maquina.id AS id_maquina,
                    %s AS PLATE_ID,
                    %s AS TIME_STAMP,
                    %s AS ROW,
                    %s AS PRED_TCBS,
                    %s AS PRED_MSA,
                    %s AS PRED_BA,
                    %s AS PRED,
                FROM maquina
                WHERE maquina.NUM_SERIE = %s
            """
        '''

        query = "INSERT INTO aquagar (PLATE_ID, TIME_STAMP, ROW,PRED_TCBS, PRED_MSA, PRED_BA, PRED, SERIAL_NUM) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        
        # ROW 1: Inserting predictions
        values = (plate_id,  timestamp , '1', json.dumps(upperRowTCBS), json.dumps(upperRowMSA), json.dumps(upperRowBA), json.dumps(upperRowPred), serial_num) 
        cursor.execute(query, values)

        # ROW 2:  Inserting predictions
        values = (plate_id,  timestamp , '2', json.dumps(lowerRowTCBS), json.dumps(lowerRowMSA), json.dumps(lowerRowBA), json.dumps(lowerRowPred), serial_num) 
        cursor.execute(query, values)

        db_connection.commit()

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        db_connection.close()

    # Would be nice to print the response
    import cv2
    img_pred = Image.fromarray(cv2.cvtColor(modelColonies(input_image)[0].plot(), cv2.COLOR_BGR2RGB))
    return StreamingResponse(content=get_bytes_from_image(img_pred), media_type="image/jpeg")

@app.get("/get_machine_predictions")
async def get_machine_predictions(id_maquina: str = Query(..., description="Select a machine_id", enum = ['1','2','3']),
                                  range: str = Query('1', description = "Select a range in days", enum = ['1', '7', '30']),
                                  ):
    import mysql.connector

    # Replace with your MySQL connection details
    host =  '10.8.0.1'
    username = 'pere'
    password = 'Nemomola5'
    database_name =  'KOAPredictions'

    # Create a connection to the MySQL server
    db_connection = mysql.connector.connect(
        host=host,
        user=username,
        password=password,
        database=database_name
    )

    # Create a cursor to execute SQL commands
    cursor = db_connection.cursor(dictionary=True)

    try:
        # Query the full table with the specified range
        query = f"SELECT * FROM aquagar WHERE STR_TO_DATE(TIME_STAMP, '%Y-%m-%d %H:%i:%s') >= DATE_SUB(CURDATE(), INTERVAL {range} DAY) AND id_maquina = {id_maquina};"

        cursor.execute(query)

        # Fetch all rows
        data = cursor.fetchall()
        cursor.close()

        return data

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        db_connection.close()

    

@app.get("/get_machine_variables")
async def get_machine_variables(topic: str = Query(..., description="Select a topic", enum = ['configuracio', 'estat', 'maquina', 'experiment']),
                                id_maquina: str = Query(..., description="Select a machine_id", enum = ['1','2','3']),
                                range: str = Query('1', description = "Select a range in days", enum = ['1', '7', '30'])
                                ):
    import mysql.connector

    # Replace with your MySQL connection details
    host =  '10.8.0.1'
    username = 'pere'
    password = 'Nemomola5'
    database_name =  'KOAMachines'

    # Create a connection to the MySQL server
    db_connection = mysql.connector.connect(
        host=host,
        user=username,
        password=password,
        database=database_name
    )

    # Create a cursor to execute SQL commands
    cursor = db_connection.cursor(dictionary=True)

    try:
        if topic == 'estat':
            # Query the full table with the specified range
            query = f"SELECT * FROM {topic} WHERE STR_TO_DATE(TIME_STAMP, '%Y-%m-%d %H:%i:%s') >= DATE_SUB(CURDATE(), INTERVAL {range} DAY) AND id_maquina = {id_maquina};"
        else: 
            # Query the full table (but range in days won't apply)
            query = f"SELECT * FROM {topic} WHERE id_maquina = {id_maquina}"

        cursor.execute(query)

        # Fetch all rows
        data = cursor.fetchall()
        cursor.close()

        return data

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        db_connection.close()