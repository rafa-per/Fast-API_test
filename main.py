from typing import Union
from typing import Annotated
from pydantic import BaseModel
from ultralytics import YOLO
import io
from PIL import Image
import json

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import FileResponse

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None


results = []

app = FastAPI()

@app.post("/files/")
async def create_file(file: Annotated[UploadFile, File(description = "A file read as bytes")]):
    """<pre>
        <b>Parameters Description:</b>
            <i>alarm</i>: a picture file in .jpg/jpeg/png format.
            <i>customerid</i>: ID of the customer.
            <i>alias</i>: an alias name for the camera/environment.
            <i>confidence</i>:  threshold confidence for YOLO detections.
            <i>label_name</i>: name that will be saved in mongodb for people detecctions.
        <br>
        <b>Example 1</b>:
            Input:
                {
                    alarm: "photo_file.jpg",
                    customerid: "0ab460db-b597-4a68-8dd8-2ccb42c31ce0",
                    alias: "camera7897998",
                    track_confidence: 35,
                    track_area: '''[{"x":10,"y":180}, {"x":800,"y":720}]''',    -
                    confidence: 30, (default=30),
                    label_name: "pessoa", (default="person")
                }
            Output (200):
                {
                    "datetime": "2023-05-05T19:38:30.906445",
                    "alias": "camera7897998",
                    "event_type": "photo-count",
                    "duration": 100.323, (time in seconds)
                    "_id": "64c07110f2e6c4dae36c38ea",
                    "summary": {
                        "pessoa": 2
                    }
                }
        <br>
        <b>Example 2</b>:
            Input:
                {
                    alarm: "invalid_file.jpg",
                    customerid: "0ab460db-b597-4a68-8dd8-2ccb42c31ce0",
                    alias: "camera7897998",
                    track_confidence: 35,
                    track_area: '''[{"x":10,"y":180}, {"x":800,"y":720}]''',    -
                    confidence: 30, (default=30),
                    label_name: "pessoa", (default="person")
                }
            Output (400):
                {
                    "detail": "encountered an issue with the input file - possible corruption detected, please refer to the logs for comprehensive technical insights"
                }
         <br>
        <b>Example 3</b>:
            Input:
                {
                    alarm: "valid_file.jpg",
                    customerid: "0ab460db-b597-4a68-8dd8-2ccb42c31ce0",
                    alias: "camera7897998",
                    track_confidence: 35,
                    track_area: '''[{"x":10222}, {"y":720}]''',    -
                    confidence: 30, (default=30),
                    label_name: "pessoa", (default="person")
                }
            Output (400):
                {
                    "detail": "you should check if the track_area was set correctly, please check the documentation accessing the url /docs and see how to fix it, for futher details search for the aplication logs"
                }
        </pre>"""
    global results
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model
    results = model(image)  # return a list of Results objects
    result_image_path = "result.jpg"
    results[0].save(filename=result_image_path)  # save to disk
    return FileResponse(result_image_path, media_type="image/jpeg")

@app.get("/result/")
async def get_results():
    global results
    return json.loads(results[0].tojson())


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/items/")
async def create_item(item: Item):
    print(item.price)
    return item


@app.get("/teste")
def teste():
    return {"rafael": "teste"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}