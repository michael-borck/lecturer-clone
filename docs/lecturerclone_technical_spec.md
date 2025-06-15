# LecturerClone - Technical Specification
## System Architecture and Implementation Guide

### System Overview

LecturerClone is a distributed system that processes multi-modal inputs (images, text, audio) through an AI pipeline to generate personalized lecture videos. The architecture emphasizes scalability, reliability, and quality control.

### High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   React Web    │    │   FastAPI        │    │   Celery Workers    │
│   Frontend      │◄──►│   Backend        │◄──►│   (AI Pipeline)     │
│                 │    │                  │    │                     │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                │                          │
                                ▼                          ▼
                       ┌──────────────────┐    ┌─────────────────────┐
                       │   PostgreSQL     │    │   File Storage      │
                       │   Database       │    │   (AWS S3/R2)       │
                       └──────────────────┘    └─────────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Redis          │
                       │   Cache/Queue    │
                       └──────────────────┘
```

### Technology Stack

#### Frontend
- **Framework**: React 18+ with TypeScript
- **Styling**: Tailwind CSS
- **State Management**: TanStack Query (React Query) + Zustand
- **File Upload**: React Dropzone
- **UI Components**: Headless UI + Custom components
- **Build Tool**: Vite
- **Testing**: Jest + React Testing Library

#### Backend API
- **Framework**: FastAPI (Python 3.11+)
- **Authentication**: JWT tokens + OAuth2
- **File Processing**: Multipart uploads with progress tracking
- **Validation**: Pydantic models
- **Documentation**: Auto-generated OpenAPI/Swagger
- **Testing**: Pytest + FastAPI TestClient

#### Task Processing
- **Queue System**: Celery with Redis broker
- **AI Pipeline**: ComfyUI workflows + custom orchestration
- **Monitoring**: Celery Flower
- **Scaling**: Kubernetes HPA for workers

#### Database
- **Primary DB**: PostgreSQL 15+
- **Caching**: Redis 7+
- **Search**: PostgreSQL full-text search (upgrade to Elasticsearch if needed)
- **Migrations**: Alembic

#### Infrastructure
- **Containerization**: Docker + Docker Compose
- **Orchestration**: Kubernetes (production) / Docker Swarm (development)
- **File Storage**: AWS S3 / Cloudflare R2
- **CDN**: CloudFlare
- **Monitoring**: Prometheus + Grafana
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana)

#### AI/ML Components
- **Framework**: ComfyUI for model orchestration
- **Models**:
  - **LLM**: Llama 3 (8B) via Ollama
  - **Image Generation**: Flux Dev + Custom LoRA
  - **Video Generation**: WAN 2.1
  - **Audio/TTS**: F5-TTS (fine-tuned)
  - **Lip Sync**: LatentSync
  - **Upscaling**: 4xLSDIR
  - **Frame Interpolation**: FILM

### System Components

#### 1. Frontend Application

**Key Features:**
- Responsive design for desktop and mobile
- Single video upload with real-time processing feedback
- Script input and validation interface
- Model training progress tracking with WebSocket updates
- Generated video preview and approval workflow
- Personal model management dashboard

**Core Pages:**
```
/dashboard          - User overview and model status
/setup             - Initial video upload and training
/create            - Script-to-video generation workflow
/models            - Personal AI model management
/history           - Generated content library
/profile           - Account settings and preferences
```

**State Management:**
```typescript
// User model state
interface UserModelState {
  isModelTrained: boolean;
  trainingProgress: number;
  modelVersion: string;
  lastTrainingDate: Date;
}

// Video processing state
interface VideoProcessingState {
  currentProject: Project | null;
  processingStatus: ProcessingStatus;
  generationAttempts: number;
  qualityScore: number;
}

// Processing status
type ProcessingStatus = 
  | 'idle' 
  | 'uploading' 
  | 'extracting_frames'
  | 'training_models'
  | 'generating_video'
  | 'post_processing'
  | 'completed' 
  | 'failed';
```

#### 2. Backend API

**Core Endpoints:**

```python
# User Model Management
POST   /api/v1/users/setup                # Initial video upload and training
GET    /api/v1/users/models               # Get user's trained models
POST   /api/v1/users/models/retrain       # Retrain models with new data
DELETE /api/v1/users/models               # Delete user models

# Video Processing
POST   /api/v1/videos/upload              # Upload training video
GET    /api/v1/videos/{id}/frames         # Get extracted frames preview
POST   /api/v1/videos/{id}/approve-frames # Approve frame selection

# Content Generation
POST   /api/v1/generate/video             # Generate video from script
GET    /api/v1/generate/{id}/status       # Get generation status
POST   /api/v1/generate/{id}/regenerate   # Regenerate with different attempt

# Content Management
GET    /api/v1/content                    # List user's generated content
GET    /api/v1/content/{id}               # Get specific video
DELETE /api/v1/content/{id}               # Delete generated video

# User Management
POST   /api/v1/auth/sso                   # University SSO authentication
GET    /api/v1/users/me                   # Get current user
PUT    /api/v1/users/me                   # Update user profile
```

**Database Schema:**

```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    university_id VARCHAR(100),
    full_name VARCHAR(255),
    department VARCHAR(255),
    is_model_trained BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- User trained models
CREATE TABLE user_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    lora_weights_path VARCHAR(500),
    voice_model_path VARCHAR(500),
    training_video_id UUID,
    model_version INTEGER DEFAULT 1,
    quality_score DECIMAL(3,2),
    training_duration_minutes INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Training videos and extracted data
CREATE TABLE training_videos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    original_video_path VARCHAR(500),
    script_text TEXT,
    extracted_frames_count INTEGER,
    audio_segments_count INTEGER,
    preprocessing_config JSONB,
    status VARCHAR(50) DEFAULT 'processing',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Extracted training frames
CREATE TABLE training_frames (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    training_video_id UUID REFERENCES training_videos(id) ON DELETE CASCADE,
    frame_path VARCHAR(500),
    processed_frame_path VARCHAR(500),
    timestamp_ms INTEGER,
    quality_score DECIMAL(3,2),
    face_detection_confidence DECIMAL(3,2),
    crop_coordinates JSONB,
    selected_for_training BOOLEAN DEFAULT FALSE
);

-- Generated content
CREATE TABLE generated_videos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    script_text TEXT,
    generation_attempt INTEGER,
    quality_score DECIMAL(3,2),
    video_path VARCHAR(500),
    duration_seconds INTEGER,
    model_version_used INTEGER,
    status VARCHAR(50), -- 'processing', 'completed', 'failed'
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### 3. AI Processing Pipeline

**Initial Training Pipeline:**

```python
@celery.app.task(bind=True)
def process_initial_training(self, user_id: str, video_path: str, script_text: str):
    """Process initial video for model training"""
    try:
        # Update status
        update_user_status(user_id, "extracting_frames")
        
        # 1. Extract and analyze frames
        raw_frames = await extract_keyframes(
            video_path,
            target_count=50,
            diversity_threshold=0.7
        )
        
        # 2. Preprocess frames (crop, enhance, focus on lecturer)
        processed_frames = []
        for frame in raw_frames:
            # Detect lecturer in frame
            face_detection = await detect_lecturer_face(frame)
            
            # Smart crop to focus on lecturer
            cropped_frame = await smart_crop_lecturer(
                frame, 
                face_detection,
                target_size=(512, 512),
                padding_ratio=0.3  # Include some background context
            )
            
            # Enhance quality
            enhanced_frame = await enhance_frame_quality(cropped_frame)
            
            # Calculate frame quality score
            quality_score = await calculate_frame_quality(enhanced_frame)
            
            if quality_score > 0.7:  # Only use high-quality frames
                processed_frames.append({
                    'frame': enhanced_frame,
                    'quality': quality_score,
                    'crop_coords': face_detection.crop_coordinates
                })
        
        # 3. Select best frames for training (20-30 diverse, high-quality frames)
        training_frames = await select_training_frames(
            processed_frames,
            target_count=25,
            quality_threshold=0.8
        )
        
        # 4. Generate captions for training frames
        captions = await generate_frame_captions(training_frames)
        
        # 5. Extract clean audio segments
        update_user_status(user_id, "processing_audio")
        audio_segments = await extract_clean_audio(
            video_path,
            script_text,
            min_duration=10,
            max_background_noise=-35dB
        )
        
        # 6. Train personalized models
        update_user_status(user_id, "training_models")
        
        # Train LoRA for image generation
        lora_task = train_lora_model.delay(training_frames, captions, user_id)
        
        # Train voice model
        voice_task = train_voice_model.delay(audio_segments, script_text, user_id)
        
        # Wait for both training tasks
        lora_result = lora_task.get()
        voice_result = voice_task.get()
        
        # 7. Save trained models
        await save_user_models(
            user_id, 
            lora_result.weights_path,
            voice_result.model_path,
            training_frames
        )
        
        # 8. Mark user as ready for generation
        await mark_user_model_ready(user_id)
        
        update_user_status(user_id, "training_completed")
        
    except Exception as exc:
        update_user_status(user_id, "training_failed")
        raise self.retry(exc=exc, countdown=300, max_retries=2)
```

**Video Generation Pipeline:**

```python
@celery.app.task(bind=True)
def generate_video_from_script(self, user_id: str, script_text: str):
    """Generate video using pre-trained user models"""
    try:
        # Load user's trained models
        user_models = await load_user_models(user_id)
        if not user_models:
            raise ValueError("User models not found or not trained")
        
        # Segment script for processing
        script_segments = await segment_script(script_text, max_duration=30)
        
        segment_results = []
        for i, segment in enumerate(script_segments):
            # Generate image using user's LoRA
            image_result = await generate_image_with_lora(
                segment.image_prompt,
                user_models.lora_weights,
                user_models.reference_style
            )
            
            # Generate audio using user's voice model
            audio_result = await generate_audio_with_voice(
                segment.text,
                user_models.voice_model
            )
            
            # Generate video segment
            video_result = await generate_video_segment(
                image_result.path,
                audio_result.path,
                segment.video_prompt
            )
            
            # Apply lip sync
            synced_result = await apply_lip_sync(
                video_result.path,
                audio_result.path
            )
            
            segment_results.append(synced_result)
        
        # Combine all segments
        final_video = await combine_video_segments(segment_results)
        
        # Post-process (upscale, interpolate)
        enhanced_video = await post_process_video(final_video.path)
        
        # Calculate quality score
        quality_score = await calculate_video_quality(enhanced_video.path)
        
        # Save result
        await save_generated_video(user_id, enhanced_video.path, quality_score)
        
        return enhanced_video.path
        
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60, max_retries=3)
```

**Video Processing and Frame Enhancement:**

```python
class VideoPreprocessingService:
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.pose_detector = MediaPipe.solutions.pose.Pose()
    
    async def extract_keyframes(self, video_path: str, target_count: int = 50) -> List[Frame]:
        """Extract diverse keyframes from lecture video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_features = []
        
        frame_interval = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / (target_count * 2))
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                # Calculate frame features for diversity
                features = await self.calculate_frame_features(frame)
                
                # Check if frame is sufficiently different from existing frames
                if self.is_frame_diverse(features, frame_features, threshold=0.3):
                    frames.append({
                        'image': frame,
                        'timestamp': frame_idx / cap.get(cv2.CAP_PROP_FPS),
                        'features': features
                    })
                    frame_features.append(features)
            
            frame_idx += 1
        
        cap.release()
        return frames[:target_count]
    
    async def smart_crop_lecturer(self, frame: np.ndarray, target_size: tuple = (512, 512)) -> dict:
        """Intelligently crop frame to focus on lecturer"""
        
        # 1. Detect faces in the frame
        faces = self.face_detector.detectMultiScale(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        if len(faces) == 0:
            # No face detected, use center crop as fallback
            return self.center_crop(frame, target_size)
        
        # 2. Select the largest face (likely the lecturer)
        main_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = main_face
        
        # 3. Detect pose to understand body positioning
        pose_results = self.pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 4. Calculate optimal crop region
        crop_region = self.calculate_optimal_crop(
            frame.shape,
            main_face,
            pose_results,
            target_size,
            padding_factor=0.4  # Include some background context
        )
        
        # 5. Crop and resize
        cropped = frame[crop_region['y1']:crop_region['y2'], 
                       crop_region['x1']:crop_region['x2']]
        resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # 6. Enhance quality
        enhanced = await self.enhance_frame_quality(resized)
        
        return {
            'processed_frame': enhanced,
            'crop_coordinates': crop_region,
            'face_confidence': len(faces),
            'quality_score': await self.assess_frame_quality(enhanced)
        }
    
    async def enhance_frame_quality(self, frame: np.ndarray) -> np.ndarray:
        """Enhance frame quality using various techniques"""
        
        # 1. Noise reduction
        denoised = cv2.bilateralFilter(frame, 9, 75, 75)
        
        # 2. Sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # 3. Contrast enhancement
        enhanced = cv2.convertScaleAbs(sharpened, alpha=1.1, beta=10)
        
        # 4. Color correction (if needed)
        # Apply histogram equalization or other color corrections
        
        return enhanced
    
    async def assess_frame_quality(self, frame: np.ndarray) -> float:
        """Assess frame quality for training suitability"""
        
        # 1. Sharpness (Laplacian variance)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Brightness assessment
        brightness = np.mean(gray)
        brightness_score = 1.0 - abs(brightness - 127) / 127  # Prefer mid-range brightness
        
        # 3. Face detection confidence
        faces = self.face_detector.detectMultiScale(gray)
        face_score = min(len(faces), 1.0)  # Prefer exactly one face
        
        # 4. Contrast assessment
        contrast = gray.std()
        contrast_score = min(contrast / 50, 1.0)  # Prefer good contrast
        
        # Weighted combination
        quality_score = (
            sharpness * 0.3 +
            brightness_score * 0.2 +
            face_score * 0.3 +
            contrast_score * 0.2
        ) / 100  # Normalize to 0-1 range
        
        return min(quality_score, 1.0)
```

#### 4. Quality Control System

**Multi-Generation Strategy:**

```python
class QualityController:
    def __init__(self, target_attempts: int = 3):
        self.target_attempts = target_attempts
    
    async def generate_with_quality_control(self, project_id: str) -> str:
        """Generate multiple versions and select the best"""
        attempts = []
        
        for attempt in range(self.target_attempts):
            try:
                result = await self.generate_single_attempt(project_id, attempt)
                quality_score = await self.calculate_quality_score(result)
                
                attempts.append({
                    'result': result,
                    'quality_score': quality_score,
                    'attempt': attempt
                })
                
            except Exception as e:
                logger.error(f"Attempt {attempt} failed: {e}")
                continue
        
        if not attempts:
            raise Exception("All generation attempts failed")
        
        # Select best result
        best_attempt = max(attempts, key=lambda x: x['quality_score'])
        
        # Save all attempts for user review
        await self.save_attempts(project_id, attempts)
        
        return best_attempt['result']
    
    async def calculate_quality_score(self, video_path: str) -> float:
        """Calculate quality score based on multiple factors"""
        scores = []
        
        # Face consistency score
        face_score = await self.analyze_face_consistency(video_path)
        scores.append(face_score * 0.4)
        
        # Lip sync accuracy
        lip_sync_score = await self.analyze_lip_sync(video_path)
        scores.append(lip_sync_score * 0.3)
        
        # Audio quality
        audio_score = await self.analyze_audio_quality(video_path)
        scores.append(audio_score * 0.2)
        
        # Visual artifacts
        visual_score = await self.analyze_visual_artifacts(video_path)
        scores.append(visual_score * 0.1)
        
        return sum(scores)
```

#### 5. File Storage and CDN

**Storage Strategy:**

```python
class FileStorageService:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.bucket_name = settings.S3_BUCKET
        self.cdn_base_url = settings.CDN_BASE_URL
    
    async def upload_file(self, file: UploadFile, project_id: str, file_type: str) -> str:
        """Upload file to S3 with appropriate path structure"""
        
        # Generate path: projects/{project_id}/{file_type}/{timestamp}_{filename}
        timestamp = int(time.time())
        safe_filename = self.sanitize_filename(file.filename)
        storage_path = f"projects/{project_id}/{file_type}/{timestamp}_{safe_filename}"
        
        # Upload to S3
        await self.s3_client.upload_fileobj(
            file.file,
            self.bucket_name,
            storage_path,
            ExtraArgs={
                'ContentType': file.content_type,
                'Metadata': {
                    'project_id': project_id,
                    'file_type': file_type,
                    'original_filename': file.filename
                }
            }
        )
        
        return storage_path
    
    def get_cdn_url(self, storage_path: str) -> str:
        """Get CDN URL for a stored file"""
        return f"{self.cdn_base_url}/{storage_path}"
```

### Deployment Architecture

#### Development Environment

```yaml
# docker-compose.yml
version: '3.8'
services:
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
    environment:
      - REACT_APP_API_URL=http://localhost:8000
  
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/lecturerclone
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
  
  celery:
    build: ./backend
    command: celery -A app.celery worker -l info
    volumes:
      - ./backend:/app
      - ./models:/models
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/lecturerclone
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
  
  comfyui:
    image: comfyui/comfyui:latest
    ports:
      - "8188:8188"
    volumes:
      - ./models:/app/models
      - ./workflows:/app/workflows
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=lecturerclone
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

#### Production Kubernetes Configuration

```yaml
# kubernetes/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: lecturerclone

---
# kubernetes/backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
  namespace: lecturerclone
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: backend
        image: lecturerclone/backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"

---
# kubernetes/celery-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: celery-workers
  namespace: lecturerclone
spec:
  replicas: 2
  selector:
    matchLabels:
      app: celery-workers
  template:
    metadata:
      labels:
        app: celery-workers
    spec:
      containers:
      - name: celery-worker
        image: lecturerclone/backend:latest
        command: ["celery", "-A", "app.celery", "worker", "-l", "info"]
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: redis-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4000m"
            nvidia.com/gpu: 1
```

### Monitoring and Observability

#### Application Metrics

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')

# Business metrics
VIDEO_GENERATION_COUNT = Counter('videos_generated_total', 'Total videos generated', ['user_tier'])
VIDEO_GENERATION_DURATION = Histogram('video_generation_duration_seconds', 'Video generation time')
QUALITY_SCORE_GAUGE = Gauge('video_quality_score', 'Average video quality score')

# Resource metrics
ACTIVE_WORKERS = Gauge('celery_active_workers', 'Number of active Celery workers')
QUEUE_LENGTH = Gauge('celery_queue_length', 'Number of tasks in queue')
```

#### Health Checks

```python
@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    checks = {
        "database": await check_database_connection(),
        "redis": await check_redis_connection(),
        "storage": await check_s3_connection(),
        "comfyui": await check_comfyui_connection(),
        "models": await check_model_availability()
    }
    
    overall_status = "healthy" if all(checks.values()) else "unhealthy"
    
    return {
        "status": overall_status,
        "checks": checks,
        "timestamp": datetime.utcnow()
    }
```

### Security Considerations

#### Authentication and Authorization

```python
# JWT token configuration
JWT_SECRET_KEY = secrets.token_urlsafe(32)
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_TIME = timedelta(hours=24)

# Role-based access control
class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    ENTERPRISE = "enterprise"

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/v1/projects/{id}/process")
@limiter.limit("5/minute")  # Limit video generation requests
async def process_video(request: Request, project_id: str):
    pass
```

#### Data Protection

```python
# File validation
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_AUDIO_TYPES = {"audio/wav", "audio/mp3", "audio/m4a"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

async def validate_file_upload(file: UploadFile) -> bool:
    """Validate uploaded files for security"""
    
    # Check file type
    if file.content_type not in ALLOWED_IMAGE_TYPES | ALLOWED_AUDIO_TYPES:
        raise HTTPException(400, "Invalid file type")
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    size = file.file.tell()
    file.file.seek(0)  # Reset
    
    if size > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large")
    
    # Scan for malware (integrate with service like VirusTotal)
    await scan_file_for_malware(file)
    
    return True
```

### Performance Optimization

#### Caching Strategy

```python
# Redis caching for expensive operations
import redis.asyncio as redis

cache = redis.Redis.from_url(settings.REDIS_URL)

@lru_cache(maxsize=100)
async def get_user_subscription(user_id: str) -> dict:
    """Cache user subscription data"""
    cache_key = f"user_subscription:{user_id}"
    
    # Try cache first
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Fetch from database
    subscription = await fetch_user_subscription(user_id)
    
    # Cache for 1 hour
    await cache.setex(cache_key, 3600, json.dumps(subscription))
    
    return subscription
```

#### Database Optimization

```sql
-- Indexes for common queries
CREATE INDEX idx_projects_user_id ON projects(user_id);
CREATE INDEX idx_projects_status ON projects(status);
CREATE INDEX idx_generated_videos_project_id ON generated_videos(project_id);
CREATE INDEX idx_project_files_project_id ON project_files(project_id);

-- Partial indexes for active projects
CREATE INDEX idx_active_projects ON projects(user_id, updated_at) 
WHERE status IN ('draft', 'processing', 'reviewing');
```

This technical specification provides a comprehensive foundation for implementing LecturerClone. The architecture is designed to be scalable, maintainable, and robust enough to handle the complexities of AI-powered video generation while providing a smooth user experience.