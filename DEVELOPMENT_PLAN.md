# LecturerClone Development Plan

## Overview
Build an MVP of LecturerClone that demonstrates AI-generated lecture videos, then iterate to production readiness.

## Hardware Setup
- **Development Machine**: 2x RTX 2060 8GB GPUs
- **Feasibility**: Confirmed suitable for MVP development
- **Multi-GPU Usage**: ComfyUI supports CUDA_VISIBLE_DEVICES for GPU selection

## MVP Approach (Week 1)

### Day 1-2: Environment Setup
- Install ComfyUI and required models
- Test individual workflows manually
- Verify GPU setup (2x RTX 2060 8GB)
- Configure multi-GPU usage if needed

### Day 3-4: Basic Integration
- Create minimal FastAPI server
- Chain ComfyUI workflows together
- File-based job tracking
- Simple script → video pipeline

### Day 5-7: Simple Interface
- Basic web UI for input/output
- Generate 30-second demo videos
- Use pre-trained MichaelPound models
- Skip user training initially

## Post-MVP Iterations

### Phase 2: Core Features (Weeks 2-3)
- Add PostgreSQL database
- Implement Redis + Celery queuing
- Basic user management
- Multi-GPU optimization
- Longer video generation

### Phase 3: Production Ready (Weeks 4-8)
- Full training pipeline
- Quality control system
- S3 file storage
- Monitoring and logging
- Docker deployment
- Authentication system
- API documentation

## Port Configuration
All ports are configurable through environment variables:

- **ComfyUI**: 8188 (via --listen flag)
- **FastAPI**: 8000 (via ENV or uvicorn params)
- **React Dev**: 3000 (via vite.config.js)
- **PostgreSQL**: 5432 (standard Docker port)
- **Redis**: 6379 (standard Docker port)
- **Celery Flower**: 5555 (monitoring UI)

## GPU Performance Expectations

### With 2x RTX 2060 8GB:
- **Image Generation**: 30-60 seconds per image
- **Audio Generation**: 10-20 seconds per minute of audio
- **Video Processing**: 1-2 minutes per 30-second clip
- **Total Pipeline**: ~3-5 minutes for 30-second video

### Optimization Strategy:
- Use GPU0 for image generation (Flux)
- Use GPU1 for audio/video processing
- Implement model caching to avoid reload
- Consider quantization for larger models

## Development Progression

### MVP (Proof of Concept)
- Single workflow execution
- Pre-trained models only
- 30-second videos
- File-based storage
- No authentication

### Beta (Feature Complete)
- User accounts
- Custom model training
- Multiple video lengths
- Database storage
- Basic queue system

### Production (Scalable)
- Full queue management
- S3 storage
- Monitoring/logging
- API rate limiting
- Docker deployment
- SSL/HTTPS
- Backup systems

## Next Steps
1. Transfer this plan to development server
2. Set up ComfyUI environment
3. Download required models:
   - Flux Dev
   - MichaelPound LoRA
   - F5-TTS
   - Llama 3.2
4. Test individual workflows
5. Begin MVP implementation

## Success Criteria for MVP
- [ ] Generate a script from text prompt
- [ ] Convert script to audio
- [ ] Generate matching images
- [ ] Create basic video with audio
- [ ] Simple web interface
- [ ] End-to-end demo working

## Notes
- Slower GPU performance is acceptable for MVP
- Focus on proving the concept works
- Optimization comes after validation
- Keep initial videos short (30 seconds)
- Use existing MichaelPound models to skip training