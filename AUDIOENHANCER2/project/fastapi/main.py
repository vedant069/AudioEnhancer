from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
from services.audio_service import AudioService
from services.video_service import VideoService
import scipy.io.wavfile as wav

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("uploads/shorts", exist_ok=True)

# Mount the shorts directory
app.mount("/shorts", StaticFiles(directory="uploads/shorts"), name="shorts")

@app.post("/enhance-audio")
async def enhance_audio(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")
            
        # Save uploaded file
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the audio
        audio_service = AudioService()
        enhanced_file_path = await audio_service.process_audio(file_path)
        
        # Return the enhanced audio file
        return FileResponse(
            enhanced_file_path,
            media_type="audio/wav",
            filename="enhanced_audio.wav"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup uploaded files
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            if os.path.exists(enhanced_file_path):
                os.remove(enhanced_file_path)
        except:
            pass

@app.post("/process-youtube")
async def process_youtube(url: str = Form(...)):
    try:
        video_service = VideoService()
        shorts = await video_service.process_youtube_video(url)
        
        if not shorts:
            raise HTTPException(status_code=500, detail="Failed to process video")
        
        # Return the paths to the generated shorts
        return {
            "shorts": [
                {
                    "url": f"/shorts/{os.path.basename(short['file'])}",
                    "script": short['script']
                }
                for short in shorts
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))