from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io

app = FastAPI()

# Load the YOLO model
model = YOLO('yolov8n.pt')

@app.get("/")
async def health_check():
    return {"message": "Welcome to the Object Detection API"}

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Perform object detection
    results = model(image)
    
    # Draw bounding boxes and labels on the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers
            class_id = int(box.cls)
            conf = float(box.conf)
            label = f"{result.names[class_id]} {conf:.2f}"
            
            # Ensure coordinates are valid
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            try:
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                
                # Draw label
                text_width = draw.textlength(label, font=font)
                draw.rectangle([x1, y1, x1 + text_width, y1 - 20], fill="red")
                draw.text((x1, y1 - 20), label, fill="white", font=font)
            except ValueError as e:
                print(f"Error drawing box: {e}")
                continue
    
    # Save the image to a bytes buffer
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return StreamingResponse(img_byte_arr, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
