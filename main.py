from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io
import logging

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the YOLO model
model = YOLO('yolov8n.pt')

@app.get("/")
async def health_check():
    return {"message": "Welcome to the Object Detection API"}

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Perform object detection
        results = model(image)
        
        # Draw bounding boxes and labels on the image
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        
        width, height = image.size
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                try:
                    print(box)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    print(x1, y1, x2, y2)
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers
                    
                    # Ensure coordinates are within image boundaries
                    x1 = max(0, min(x1, width - 1))
                    x2 = max(0, min(x2, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    y2 = max(0, min(y2, height - 1))
                    
                    # Ensure x1 < x2 and y1 < y2 (rectangles must have non-zero dimensions)
                    if x1 >= x2 or y1 >= y2:
                        logger.warning(f"Skipping invalid box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                        continue
                    
                    class_id = int(box.cls)
                    conf = float(box.conf)
                    label = f"{result.names[class_id]} {conf:.2f}"
                    
                    # Draw bounding box
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                    
                    # Draw label
                    text_width = draw.textlength(label, font=font)
                    text_height = font.size
                    draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill="red")
                    draw.text((x1, y1 - text_height), label, fill="white", font=font)
                    
                except Exception as e:
                    logger.error(f"Error processing box: {e}")
                    continue

        
        # Save the image to a bytes buffer
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/png")
    
    except Exception as e:
        logger.error(f"Error in detect_objects: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)