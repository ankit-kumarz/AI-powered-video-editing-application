from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import sys
import uuid
import shutil
from pathlib import Path
import subprocess
import aiofiles
from typing import List, Optional
import json
import time
from fastapi import Request

# Add current directory to Python path for service imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="AI Video Editor API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
UPLOAD_DIR = Path("temp/uploads")
PROCESSED_DIR = Path("temp/processed")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files for downloads
app.mount("/downloads", StaticFiles(directory="temp/processed"), name="downloads")

# Store processing status
processing_status = {}

# Lazy import services
def get_service(service_name):
    """Lazy import services to avoid startup issues"""
    if service_name == "background_removal":
        from services.background_removal import BackgroundRemovalService
        return BackgroundRemovalService()
    elif service_name == "subtitles":
        from services.subtitles import SubtitleService
        return SubtitleService()
    elif service_name == "scene_detection":
        from services.scene_detection import SceneDetectionService
        return SceneDetectionService()
    elif service_name == "voice_translate":
        from services.voice_translate_optimized import OptimizedVoiceTranslationService
        return OptimizedVoiceTranslationService()
    elif service_name == "style_filters":
        from services.style_filters import StyleFilterService
        return StyleFilterService()
    elif service_name == "object_removal":
        from services.object_removal import ObjectRemovalService
        return ObjectRemovalService()
    elif service_name == "auto_cut_silence":
        from services.auto_cut_silence import AutoCutSilenceService
        return AutoCutSilenceService()
    elif service_name == "video_compilation":
        from services.video_compilation import VideoCompilationService
        return VideoCompilationService()
    elif service_name == "ai_editing_suggestions":
        from services.ai_editing import AIEditingSuggestionsService
        return AIEditingSuggestionsService()
    elif service_name == "script_analysis":
        from services.script_analysis import ScriptAnalysisService
        return ScriptAnalysisService()
    else:
        raise ValueError(f"Unknown service: {service_name}")

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload video file for processing"""
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    # Generate unique ID for this upload
    upload_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{upload_id}_{file.filename}"
    
    try:
        content = await file.read()
        # Enforce max size 500MB
        if len(content) > 500 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 500MB)")
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        # Enforce max duration 10 minutes
        try:
            duration = await _probe_duration_seconds(str(file_path))
            if duration is not None and duration > 10 * 60:
                file_path.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail="Video too long (max 10 minutes)")
        except HTTPException:
            raise
        except Exception:
            pass

        # Initialize processing status
        processing_status[upload_id] = {
            "status": "uploaded",
            "file_path": str(file_path),
            "filename": file.filename,
            "progress": 0
        }
        
        return {
            "upload_id": upload_id,
            "filename": file.filename,
            "message": "Video uploaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def _probe_duration_seconds(path: str) -> float | None:
    """Return duration in seconds using ffprobe, or None if unavailable."""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=nw=1:nk=1', path
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        duration_str = result.stdout.strip()
        return float(duration_str)
    except Exception:
        return None

@app.post("/process/background-removal")
async def process_background_removal(
    upload_id: str = Query(..., description="Upload ID"),
    background_tasks: BackgroundTasks = None,
    replace_with_image: Optional[UploadFile] = File(None),
    blur_background: bool = False
):
    """Process background removal/replacement"""
    if upload_id not in processing_status:
        raise HTTPException(status_code=404, detail="Upload ID not found")
    
    processing_status[upload_id]["status"] = "processing"
    processing_status[upload_id]["progress"] = 10

    # Persist background image (if any) to a temp path so it survives after request ends
    replace_image_path = None
    if replace_with_image is not None:
        img_suffix = Path(replace_with_image.filename).suffix or ".png"
        img_temp = Path("temp/uploads") / f"{upload_id}_bg{img_suffix}"
        async with aiofiles.open(img_temp, "wb") as f:
            content = await replace_with_image.read()
            await f.write(content)
        replace_image_path = str(img_temp)

    if background_tasks is None:
        background_tasks = BackgroundTasks()

    # Start background processing
    background_tasks.add_task(
        _process_background_removal,
        upload_id,
        processing_status,
        replace_image_path,
        blur_background
    )
    
    return {"message": "Background removal processing started", "upload_id": upload_id}

async def _process_background_removal(upload_id, processing_status, replace_image_path, blur_background):
    """Background task for background removal"""
    service = get_service("background_removal")
    await service.process(upload_id, processing_status, replace_image_path, blur_background)

@app.post("/process/subtitles")
async def process_subtitles(
    upload_id: str = Query(..., description="Upload ID"),
    background_tasks: BackgroundTasks = None,
    burn_in: bool = True,
    language: str = "en"
):
    """Generate subtitles and optionally burn them into video"""
    if upload_id not in processing_status:
        raise HTTPException(status_code=404, detail="Upload ID not found")
    
    processing_status[upload_id]["status"] = "processing"
    processing_status[upload_id]["progress"] = 10
    
    if background_tasks is None:
        background_tasks = BackgroundTasks()
    background_tasks.add_task(
        _process_subtitles,
        upload_id,
        processing_status,
        burn_in,
        language
    )
    
    return {"message": "Subtitle processing started", "upload_id": upload_id}

async def _process_subtitles(upload_id, processing_status, burn_in, language):
    """Background task for subtitle processing"""
    service = get_service("subtitles")
    await service.process(upload_id, processing_status, burn_in, language)

@app.post("/process/auto-cut-silence")
async def process_auto_cut_silence(
    upload_id: str = Query(..., description="Upload ID"),
    background_tasks: BackgroundTasks = None,
    silence_threshold_db: int = -40,
    min_silence_len_ms: int = 2000,
    enable_ai_cuts: bool = True,
):
    """Detect silent parts (> min_silence_len_ms) and cut them out with AI-powered optimal cuts and transitions."""
    if upload_id not in processing_status:
        raise HTTPException(status_code=404, detail="Upload ID not found")

    processing_status[upload_id]["status"] = "processing"
    processing_status[upload_id]["progress"] = 10

    if background_tasks is None:
        background_tasks = BackgroundTasks()

    background_tasks.add_task(
        _process_auto_cut_silence,
        upload_id,
        processing_status,
        silence_threshold_db,
        min_silence_len_ms,
        enable_ai_cuts,
    )

    return {"message": "AI-powered auto cut processing started", "upload_id": upload_id}

async def _process_auto_cut_silence(upload_id, processing_status, silence_threshold_db, min_silence_len_ms, enable_ai_cuts):
    """Background task for auto cut silence"""
    service = get_service("auto_cut_silence")
    await service.process(upload_id, processing_status, silence_threshold_db, min_silence_len_ms, enable_ai_cuts)

@app.post("/process/scene-detection")
async def process_scene_detection(
    upload_id: str = Query(..., description="Upload ID"),
    background_tasks: BackgroundTasks = None,
    threshold: float = 30.0
):
    """Detect scene changes and split video into separate clips"""
    if upload_id not in processing_status:
        raise HTTPException(status_code=404, detail="Upload ID not found")
    
    processing_status[upload_id]["status"] = "processing"
    processing_status[upload_id]["progress"] = 10
    
    if background_tasks is None:
        background_tasks = BackgroundTasks()
    background_tasks.add_task(
        _process_scene_detection,
        upload_id,
        processing_status,
        threshold
    )
    
    return {"message": "Scene detection processing started", "upload_id": upload_id}

async def _process_scene_detection(upload_id, processing_status, threshold):
    """Background task for scene detection"""
    service = get_service("scene_detection")
    await service.process(upload_id, processing_status, threshold)

@app.post("/process/voice-translate")
async def process_voice_translation(
    upload_id: str = Query(..., description="Upload ID"),
    background_tasks: BackgroundTasks = None,
    target_language: str = "es",
    voice_type: str = "female",
    add_subtitles: bool = True
):
    """Translate speech and generate new audio with optional subtitles"""
    if upload_id not in processing_status:
        raise HTTPException(status_code=404, detail="Upload ID not found")
    
    processing_status[upload_id]["status"] = "processing"
    processing_status[upload_id]["progress"] = 5
    processing_status[upload_id]["message"] = "Starting optimized translation..."
    
    if background_tasks is None:
        background_tasks = BackgroundTasks()
    background_tasks.add_task(
        _process_voice_translation,
        upload_id,
        processing_status,
        target_language,
        voice_type,
        add_subtitles
    )
    
    return {"message": "Optimized voice translation processing started", "upload_id": upload_id}

async def _process_voice_translation(upload_id, processing_status, target_language, voice_type, add_subtitles):
    """Background task for voice translation"""
    service = get_service("voice_translate")
    await service.process(upload_id, processing_status, target_language, voice_type, add_subtitles)

@app.post("/process/style")
async def process_style_filters(
    upload_id: str = Query(..., description="Upload ID"),
    background_tasks: BackgroundTasks = None,
    style: str = "cinematic",
    intensity: float = 0.5
):
    """Apply artistic style filters to video"""
    if upload_id not in processing_status:
        raise HTTPException(status_code=404, detail="Upload ID not found")
    
    processing_status[upload_id]["status"] = "processing"
    processing_status[upload_id]["progress"] = 10
    
    if background_tasks is None:
        background_tasks = BackgroundTasks()
    background_tasks.add_task(
        _process_style_filters,
        upload_id,
        processing_status,
        style,
        intensity
    )
    
    return {"message": "Style filter processing started", "upload_id": upload_id}

async def _process_style_filters(upload_id, processing_status, style, intensity):
    """Background task for style filters"""
    service = get_service("style_filters")
    await service.process(upload_id, processing_status, style)

@app.post("/process/object-remove")
async def process_object_removal(
    upload_id: str = Query(..., description="Upload ID"),
    bounding_boxes: str = Query(None, description="Bounding boxes in format 'x1,y1,x2,y2;x1,y1,x2,y2'"),
    background_tasks: BackgroundTasks = None
):
    """Remove unwanted objects from video"""
    if upload_id not in processing_status:
        raise HTTPException(status_code=404, detail="Upload ID not found")
    
    processing_status[upload_id]["status"] = "processing"
    processing_status[upload_id]["progress"] = 10
    
    if background_tasks is None:
        background_tasks = BackgroundTasks()
    background_tasks.add_task(
        _process_object_removal,
        upload_id,
        processing_status,
        bounding_boxes
    )
    
    return {"message": "Object removal processing started", "upload_id": upload_id}

async def _process_object_removal(upload_id, processing_status, bounding_boxes):
    """Background task for object removal"""
    service = get_service("object_removal")
    await service.process(upload_id, processing_status, bounding_boxes)

@app.post("/process/video-compilation")
async def process_video_compilation(
    upload_ids: str = Query(..., description="Comma-separated list of upload IDs (max 5)"),
    background_tasks: BackgroundTasks = None,
    max_duration: int = 300,
    transition_style: str = "fade",
    preset: str = "youtube_shorts",
    apply_effects: bool = False,
    effect_type: str = "none"
):
    """Create compilation video from multiple uploaded videos"""
    # Parse comma-separated upload IDs
    upload_id_list = [uid.strip() for uid in upload_ids.split(',') if uid.strip()]
    
    if len(upload_id_list) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 videos allowed")
    
    # Validate all upload IDs exist
    for upload_id in upload_id_list:
        if upload_id not in processing_status:
            raise HTTPException(status_code=404, detail=f"Upload ID {upload_id} not found")
    
    # Use first upload ID as main for status tracking
    main_upload_id = upload_id_list[0]
    processing_status[main_upload_id]["status"] = "processing"
    processing_status[main_upload_id]["progress"] = 10
    
    if background_tasks is None:
        background_tasks = BackgroundTasks()
    background_tasks.add_task(
        _process_video_compilation,
        upload_id_list,
        processing_status,
        max_duration,
        transition_style,
        preset,
        apply_effects,
        effect_type
    )
    
    return {"message": "Video compilation processing started", "main_upload_id": main_upload_id}

async def _process_video_compilation(upload_ids, processing_status, max_duration, transition_style, preset, apply_effects, effect_type):
    """Background task for video compilation"""
    service = get_service("video_compilation")
    await service.process(upload_ids, processing_status, max_duration, transition_style, preset, apply_effects, effect_type)

@app.get("/status/{upload_id}")
async def get_processing_status(upload_id: str):
    """Get processing status for an upload"""
    if upload_id not in processing_status:
        raise HTTPException(status_code=404, detail="Upload ID not found")
    
    return processing_status[upload_id]

@app.get("/download/by-upload/{upload_id}")
async def download_processed_by_upload(upload_id: str):
    """Download the first processed file for an upload and clean it up after sending."""
    # Check both processed and compilation directories for files
    processed_files = list(PROCESSED_DIR.glob(f"{upload_id}*"))
    compilation_files = list(Path("temp/compilation").glob(f"{upload_id}*"))
    
    # Combine and sort by modification time (newest first)
    all_files = processed_files + compilation_files
    if not all_files:
        raise HTTPException(status_code=404, detail="Processed file not found")

    # Get the most recent file
    file_path = max(all_files, key=lambda f: f.stat().st_mtime)

    from starlette.background import BackgroundTask
    def _cleanup():
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass

    return FileResponse(path=file_path, filename=file_path.name, media_type="video/mp4", background=BackgroundTask(_cleanup))

@app.get("/download/{file_id}")
async def download_by_file_id(file_id: str):
    """Download a processed file by filename and clean it up after download."""
    file_path = PROCESSED_DIR / file_id
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Schedule cleanup of the file after response is sent
    from starlette.background import BackgroundTask
    def _cleanup():
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass

    return FileResponse(path=file_path, filename=file_path.name, media_type="video/mp4", background=BackgroundTask(_cleanup))

@app.get("/presets/compilation")
async def get_compilation_presets():
    """Get available compilation presets for different social media platforms"""
    service = get_service("video_compilation")
    return service.get_compilation_presets()

@app.get("/effects/post-compilation")
async def get_post_compilation_effects():
    """Get available post-compilation effects"""
    service = get_service("video_compilation")
    return service.get_post_compilation_effects()

@app.get("/presets/transitions")
async def get_transition_styles():
    """Get available transition styles"""
    service = get_service("video_compilation")
    return service.get_transition_styles()

@app.get("/downloads/{upload_id}")
async def list_downloads(upload_id: str):
    """List all available downloads for an upload (useful for scene detection)"""
    if upload_id not in processing_status:
        raise HTTPException(status_code=404, detail="Upload ID not found")
    
    processed_files = list(PROCESSED_DIR.glob(f"{upload_id}*"))
    compilation_files = list(Path("temp/compilation").glob(f"{upload_id}*"))
    all_files = processed_files + compilation_files
    
    return {
        "upload_id": upload_id,
        "files": [
            {
                "filename": f.name,
                "download_url": f"/downloads/{f.name}",
                "size": f.stat().st_size
            }
            for f in all_files
        ]
    }

@app.get("/video/{upload_id}")
async def get_video_stream(upload_id: str):
    """Stream video file for preview (doesn't delete the file)"""
    # Check both processed and compilation directories for files
    processed_files = list(PROCESSED_DIR.glob(f"{upload_id}*"))
    compilation_files = list(Path("temp/compilation").glob(f"{upload_id}*"))
    
    # Combine and sort by modification time (newest first)
    all_files = processed_files + compilation_files
    if not all_files:
        raise HTTPException(status_code=404, detail="Video file not found")

    # Get the most recent file
    file_path = max(all_files, key=lambda f: f.stat().st_mtime)

    return FileResponse(path=file_path, filename=file_path.name, media_type="video/mp4")

@app.get("/debug/status")
async def debug_status():
    """Debug endpoint to see current processing status"""
    return {
        "active_uploads": list(processing_status.keys()),
        "upload_count": len(processing_status),
        "processing_status": processing_status
    }

@app.delete("/cleanup/{upload_id}")
async def cleanup_files(upload_id: str):
    """Clean up temporary files for an upload"""
    files_cleaned = 0
    
    try:
        # Remove uploaded file if it exists in processing_status
        if upload_id in processing_status:
            file_path = Path(processing_status[upload_id]["file_path"])
            if file_path.exists():
                file_path.unlink()
                files_cleaned += 1
            
            # Remove from status
            del processing_status[upload_id]
        
        # Remove processed files (regardless of processing_status)
        processed_files = list(PROCESSED_DIR.glob(f"{upload_id}*"))
        for f in processed_files:
            try:
                f.unlink()
                files_cleaned += 1
            except Exception as e:
                print(f"Failed to remove processed file {f}: {e}")
        
        # Remove uploaded files (regardless of processing_status)
        uploaded_files = list(UPLOAD_DIR.glob(f"{upload_id}*"))
        for f in uploaded_files:
            try:
                f.unlink()
                files_cleaned += 1
            except Exception as e:
                print(f"Failed to remove uploaded file {f}: {e}")
        
        if files_cleaned > 0:
            return {"message": f"Files cleaned up successfully ({files_cleaned} files removed)"}
        else:
            return {"message": "No files found to clean up"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@app.post("/detect-objects")
async def detect_objects(request: Request):
    """Detect objects in uploaded image and return bounding boxes"""
    try:
        # Get uploaded image
        form = await request.form()
        image_file = form.get("image")
        
        if not image_file:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Save image temporarily
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        image_path = temp_dir / f"detection_{int(time.time())}.jpg"
        with open(image_path, "wb") as f:
            f.write(image_file.file.read())
        
        # Use object removal service to detect objects
        from services.object_removal import ObjectRemovalService
        service = ObjectRemovalService()
        
        # Read image with OpenCV
        import cv2
        frame = cv2.imread(str(image_path))
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Detect objects
        detected_objects = service.detect_objects(frame)
        
        # Clean up temp file
        image_path.unlink(missing_ok=True)
        
        return detected_objects
        
    except Exception as e:
        print(f"Object detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-ai-editing")
async def process_ai_editing(request: Request):
    """Process AI editing suggestions"""
    try:
        data = await request.json()
        upload_id = data.get("upload_id")
        features = data.get("features", [])
        script_content = data.get("script_content")
        
        if not upload_id:
            raise HTTPException(status_code=400, detail="upload_id is required")
        
        # Find the uploaded video file
        video_files = list(UPLOAD_DIR.glob(f"{upload_id}*"))
        if not video_files:
            raise HTTPException(status_code=404, detail="Video file not found")
        
        video_path = str(video_files[0])
        
        # Initialize services based on requested features
        suggestions = []
        video_features = []
        
        if "ai_editing_suggestions" in features:
            ai_service = get_service("ai_editing_suggestions")
            suggestions = await ai_service.generate_editing_suggestions(video_path, script_content or "")
            video_features = await ai_service.analyze_video_features(video_path)
        
        if "script_analysis" in features and script_content:
            script_service = get_service("script_analysis")
            script_suggestions = script_service.analyze_script(script_content)
            suggestions.extend(script_suggestions)
        
        if "subtitle_generation" in features and script_content:
            subtitle_service = get_service("subtitles")
            # Generate subtitles from script
            srt_path = subtitle_service.generate_subtitles_from_text(script_content, video_path)
            # Burn subtitles into video
            output_path = PROCESSED_DIR / f"{upload_id}_with_subtitles.mp4"
            result_path = subtitle_service.burn_subtitles(video_path, srt_path, str(output_path))
            print(f"✅ Subtitle generation completed: {result_path}")
        
        # Count suggestions by type
        cut_suggestions = len([s for s in suggestions if hasattr(s, 'suggestion_type') and s.suggestion_type == "cut"])
        transition_suggestions = len([s for s in suggestions if hasattr(s, 'suggestion_type') and s.suggestion_type == "transition"])
        
        return {
            "success": True,
            "suggestions": [
                {
                    "timestamp": s.timestamp,
                    "suggestion_type": s.suggestion_type,
                    "description": getattr(s, 'description', s.reason),  # Use reason as description if description doesn't exist
                    "confidence": s.confidence,
                    "reasoning": getattr(s, 'reasoning', s.reason),  # Use reason as reasoning if reasoning doesn't exist
                    "metadata": getattr(s, 'metadata', {})
                }
                for s in suggestions
            ],
            "video_features": [
                {
                    "timestamp": f.timestamp,
                    "feature_type": f.feature_type,
                    "confidence": f.confidence,
                    "description": getattr(f, 'description', f.feature_type),  # Use feature_type as description if description doesn't exist
                    "metadata": getattr(f, 'metadata', {})
                }
                for f in video_features
            ],
            "total_suggestions": len(suggestions),
            "cut_suggestions": cut_suggestions,
            "transition_suggestions": transition_suggestions
        }
        
    except Exception as e:
        print(f"AI editing error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
