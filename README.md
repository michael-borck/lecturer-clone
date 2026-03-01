# LecturerClone

<!-- BADGES:START -->
[![ai](https://img.shields.io/badge/-ai-ff6f00?style=flat-square)](https://github.com/topics/ai) [![education-technology](https://img.shields.io/badge/-education--technology-blue?style=flat-square)](https://github.com/topics/education-technology) [![fastapi](https://img.shields.io/badge/-fastapi-009688?style=flat-square)](https://github.com/topics/fastapi) [![media-production](https://img.shields.io/badge/-media--production-blue?style=flat-square)](https://github.com/topics/media-production) [![python](https://img.shields.io/badge/-python-3776ab?style=flat-square)](https://github.com/topics/python) [![react](https://img.shields.io/badge/-react-61dafb?style=flat-square)](https://github.com/topics/react) [![text-to-video](https://img.shields.io/badge/-text--to--video-blue?style=flat-square)](https://github.com/topics/text-to-video) [![video-generation](https://img.shields.io/badge/-video--generation-blue?style=flat-square)](https://github.com/topics/video-generation) [![voice-cloning](https://img.shields.io/badge/-voice--cloning-blue?style=flat-square)](https://github.com/topics/voice-cloning) [![edtech](https://img.shields.io/badge/-edtech-4caf50?style=flat-square)](https://github.com/topics/edtech)
<!-- BADGES:END -->
## Your expertise, everywhere

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![React](https://img.shields.io/badge/react-%2320232a.svg?style=flat&logo=react&logoColor=%2361DAFB)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)

LecturerClone is an AI-powered platform that enables educators to create personalized video lectures by uploading photos, scripts, and voice samples. Generate professional educational content that maintains your teaching presence while scaling to unlimited audiences.

🎯 **Perfect for**: University professors, online course creators, corporate trainers, and educational institutions

## ✨ Features

### 🎬 AI-Powered Video Generation
- **One-Time Setup**: Upload a single lecture video with script to train your personal AI model
- **Intelligent Frame Processing**: Automatic extraction and lecturer-focused cropping of training frames
- **Voice Cloning**: Extract and replicate your speaking style from the source video
- **Script-to-Video Creation**: Generate unlimited new content using only text scripts
- **Professional Quality**: HD output with lip-sync accuracy and smooth motion

### 🔄 Quality Control System
- **Multi-Generation Approach**: Creates multiple versions and selects the best result
- **Automated Quality Scoring**: Evaluates face consistency, lip-sync accuracy, and visual quality
- **Preview & Approval Workflow**: Review generated content before finalizing
- **Segmented Processing**: Handles long-form content by breaking into manageable segments

### 🎛️ User-Friendly Interface
- **Drag-and-Drop Uploads**: Intuitive file management for images, audio, and scripts
- **Real-Time Progress Tracking**: Live updates during video processing
- **Batch Processing**: Queue multiple videos for efficient workflow
- **Template Library**: Pre-built educational formats and styles

### 🏢 Enterprise Ready
- **Scalable Architecture**: Built on FastAPI, React, and Kubernetes
- **API Access**: Integrate with existing educational platforms
- **Multi-User Support**: Team collaboration and institutional accounts
- **Analytics Dashboard**: Track usage, quality metrics, and performance

## 🚀 Quick Start

### Prerequisites
- **Hardware**: NVIDIA GPU (RTX 30 series or higher recommended)
- **Software**: Docker & Docker Compose
- **Storage**: 50GB+ free space for AI models
- **Memory**: 16GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/michael-borck/lecture-clone.git
cd lecture-clone
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Download AI models** (automated script)
```bash
./scripts/download-models.sh
```

4. **Start the application**
```bash
docker-compose up -d
```

5. **Access the application**
- Frontend: http://localhost:3000
- API Documentation: http://localhost:8000/docs
- ComfyUI Interface: http://localhost:8188

### First Video Creation

1. **Create an account** at http://localhost:3000
2. **Upload your training video**:
   - Single lecture recording (10-30 minutes recommended)
   - Matching script or transcript
   - Ensure lecturer fills most of the frame for best results
3. **Wait for model training** (1-2 hours for personalized AI models)
4. **Create new content**:
   - Write your script for new video
   - Generate using your trained models
   - Preview and download your content

### Optimal Training Video Guidelines

**Video Quality:**
- **Lecturer prominence**: Ensure you fill 60-80% of the frame
- **Good lighting**: Even, front-facing illumination
- **Stable camera**: Minimal movement or shaking
- **Clear audio**: Minimal background noise
- **Multiple poses**: Natural gestures and head movements

**Content Requirements:**
- **Duration**: 10-30 minutes of speaking time
- **Variety**: Different expressions and gestures
- **Consistent environment**: Similar to desired output setting
- **Matching script**: Accurate transcript for audio alignment

## 🏗️ Architecture

LecturerClone uses a modular, scalable architecture:

```
React Frontend ↔ FastAPI Backend ↔ Celery Workers ↔ AI Pipeline (ComfyUI)
                       ↕                    ↕
                  PostgreSQL          File Storage (S3)
                       ↕
                    Redis Cache
```

### Core Technologies

**Frontend Stack:**
- React 18+ with TypeScript
- Tailwind CSS for styling
- TanStack Query for state management
- Real-time WebSocket updates

**Backend Stack:**
- FastAPI (Python 3.11+)
- PostgreSQL database
- Redis for caching and queuing
- Celery for background processing

**AI Pipeline:**
- **ComfyUI**: Model orchestration framework
- **Llama 3**: Script processing and scene generation
- **Flux + LoRA**: Personalized image generation from extracted frames
- **F5-TTS**: Voice cloning from video audio extraction
- **WAN 2.1**: Image-to-video generation
- **LatentSync**: Lip synchronization
- **FILM/4xLSDIR**: Quality enhancement and frame interpolation
- **OpenCV/MediaPipe**: Video preprocessing and lecturer detection

## 🏛️ University Service

LecturerClone is designed as an internal university service to support faculty and educational staff:

### 🎓 Faculty Access
- **Unlimited video generation** for academic staff
- **Multiple quality attempts** to ensure best results
- **HD/4K output** for professional lecture content
- **Batch processing** for course development
- **Priority support** through IT services

### 🔧 Technical Features
- **Single Sign-On (SSO)** integration with university systems
- **Course management** integration (Moodle, Canvas, Blackboard)
- **Institutional storage** with university data policies
- **Access controls** based on faculty roles and permissions
- **Usage analytics** for resource planning and optimization

## 🔧 Development

### Local Development Setup

1. **Backend development**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

2. **Frontend development**
```bash
cd frontend
npm install
npm run dev
```

3. **AI Pipeline setup**
```bash
cd ai-pipeline
python setup_comfyui.py
# Follow ComfyUI installation guide
```

### Running Tests
```bash
# Backend tests
cd backend && pytest

# Frontend tests  
cd frontend && npm test

# Integration tests
docker-compose -f docker-compose.test.yml up
```

## 📚 Documentation

### User Guides
- [Getting Started Guide](docs/getting-started.md)
- [Best Practices for Training Data](docs/training-data-guide.md)
- [Video Quality Optimization](docs/quality-guide.md)
- [API Documentation](docs/api.md)

### Technical Docs
- [Architecture Overview](docs/architecture.md)
- [AI Pipeline Details](docs/ai-pipeline.md)
- [Deployment Guide](docs/deployment.md)
- [Contributing Guidelines](CONTRIBUTING.md)

## 🔐 Privacy & Ethics

LecturerClone is built with responsible AI principles:

### Data Protection
- **User Consent**: Explicit permission required for likeness usage
- **Secure Storage**: Encrypted file storage and processing
- **GDPR Compliance**: Right to deletion and data portability
- **Educational Focus**: Designed specifically for legitimate educational use

### Ethical Guidelines
- **Clear AI Labeling**: All generated content marked as AI-created
- **Usage Policies**: Strict terms preventing misuse
- **Consent Verification**: Multi-step approval process
- **Transparency**: Open about AI limitations and capabilities

### Content Policies
- ✅ Educational lectures and tutorials
- ✅ Corporate training materials
- ✅ Language learning content
- ❌ Misleading or deceptive content
- ❌ Unauthorized use of others' likeness
- ❌ Harmful or inappropriate material

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Areas
- **AI Model Integration**: Adding new models and improving quality
- **User Experience**: Frontend improvements and workflow optimization
- **Performance**: Scaling and optimization improvements
- **Documentation**: Guides, tutorials, and API documentation

### Getting Involved
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📊 Project Status

### Current Status: **Beta Release**
- ✅ Core AI pipeline functional
- ✅ Web interface complete
- ✅ Quality control system implemented
- ✅ Docker deployment ready
- 🔄 Kubernetes production deployment (in progress)
- 🔄 Advanced analytics dashboard (coming soon)
- 🔄 Mobile app (planned)

### Roadmap
- **Q2 2025**: Production deployment, mobile app
- **Q3 2025**: Advanced personalization, multi-language support
- **Q4 2025**: Real-time generation, VR integration

## ⚠️ Important Disclaimers

### Quality Expectations
Like all AI systems, LecturerClone has limitations:
- **Training video quality matters**: Better source material yields better results
- **Frame composition is critical**: Lecturer should fill 60-80% of frame for optimal training
- **Results vary**: Quality depends on input video quality and script complexity
- **Processing time**: Initial training takes 1-2 hours, subsequent videos 10-30 minutes
- **Cherry-picked examples**: Marketing materials show best-case scenarios
- **Hardware requirements**: Significant GPU resources needed for optimal performance

### Responsible Use
- **Educational focus**: Designed for legitimate educational purposes
- **Consent required**: Only use your own likeness or with explicit permission
- **Transparency**: Always disclose when content is AI-generated
- **Legal compliance**: Follow local laws regarding AI-generated content

## 🆘 Support

### Community Support
- [GitHub Discussions](https://github.com/michael-borck/lecture-clone/discussions)
- [Discord Community](https://discord.gg/lecturerclone)
- [Reddit Community](https://reddit.com/r/lecturerclone)

### Professional Support
- **Email**: support@lecturerclone.com
- **Documentation**: [docs.lecturerclone.com](https://docs.lecturerclone.com)
- **Status Page**: [status.lecturerclone.com](https://status.lecturerclone.com)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

LecturerClone builds upon amazing open-source technologies:
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - AI workflow framework
- [Flux](https://github.com/black-forest-labs/flux) - Image generation
- [F5-TTS](https://github.com/SWivid/F5-TTS) - Text-to-speech synthesis
- [LatentSync](https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper) - Lip synchronization
- Original inspiration from [MikeBot](https://github.com/lewismorton) by Lewis Morton

Special thanks to the AI research community for making these technologies accessible and open-source.

---

**Made with ❤️ for educators worldwide**

*Transform your teaching. Scale your impact. Your expertise, everywhere.*