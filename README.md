# AI Video Editor 🎬
 
A powerful, full-stack AI-powered video editing application that combines cutting-edge machine learning technologies to transform your videos with minimal effort. Built with modern web technologies and advanced AI services. 
## 🌟 Features 

### 🎯 Core AI Features
- **🤖 AI Video Optimization**: Complete end-to-end AI pipeline with scene analysis, silence removal, subtitle generation, and quality optimization
- **✂️ AI Film Editing Suggestions**: Intelligent editing recommendations with cuts, transitions, emphasis, and pace changes based on video analysis
- **📝 AI Script Analysis**: Advanced script-based editing suggestions with emotion detection, speaker analysis, and transition indicators
- **🎬 Interactive Timeline**: Visual timeline with colored markers for different editing suggestions and click-to-navigate functionality
- **✂️ AI Auto-Cut & Transitions**: Intelligent scene-aware cuts with smooth transitions and smart merging
- **🎭 AI Background Removal**: Advanced background removal with replacement options using Rembg
- **📝 AI Subtitle Generation**: Multi-language speech-to-text with SRT generation using OpenAI Whisper
- **🎬 AI Scene Detection**: Automatic video splitting with individual download links
- **🌍 Voice Translation & Dubbing**: Complete video localization with speech recognition, translation, AI voice generation, and subtitle translation
- **🎨 AI Style Filters**: Neural style transfer with multiple artistic styles
- **🚫 AI Object Removal**: Bounding box selection with advanced inpainting technology (3-4x faster with parallel processing)
- **⚡ Optimized Video Compilation**: Parallel processing for compiling up to 5 videos efficiently with 3x faster performance

### 🎨 User Experience
- **Modern Web Interface**: Beautiful, responsive UI built with Next.js and Tailwind CSS
- **Interactive Timeline Navigation**: Click on timeline dots to jump to specific editing suggestions
- **Real-time Processing**: Live progress updates and status monitoring
- **Drag & Drop Upload**: Intuitive video upload with format validation
- **Preview Functionality**: Video preview before and after processing
- **Batch Processing**: Process multiple features simultaneously 
- **Suggestion Filtering**: Filter suggestions by type (cuts, transitions, emphasis, pace changes)
- **Compact Mode**: Toggle between detailed and compact suggestion views

## 🏗️ Project Architecture

### 📁 File Structure
```
videoeditor/
├── backend/                    # FastAPI backend server
│   ├── main.py                # Main API server and endpoints
│   ├── services/              # AI processing services 
│   │   ├── ai_editing_suggestions.py  # AI editing suggestions service
│   │   ├── script_analysis.py         # Script analysis service
│   │   ├── subtitles_simple.py        # Simplified subtitle service
│   │   ├── auto_cut_silence.py
│   │   ├── background_removal.py
│   │   ├── object_removal.py
│   │   ├── scene_detection.py
│   │   ├── style_filters.py
│   │   ├── subtitles.py
│   │   ├── video_compilation.py
│   │   ├── voice_translate.py
│   │   └── voice_translate_optimized.py 
│   ├── requirements.txt       # Python dependencies
│   └── temp/                  # Temporary file storage
├── frontend/                  # Next.js frontend application
│   ├── app/                   # Next.js 14 app directory
│   │   ├── components/        # React components
│   │   │   ├── AIEditingSelector.tsx      # AI editing feature selector
│   │   │   ├── AIEditingSuggestions.tsx   # Interactive suggestions display
│   │   │   ├── ScriptInput.tsx            # Script input component
│   │   │   ├── FeatureSelector.tsx
│   │   │   ├── ProcessingStatus.tsx 
│   │   │   ├── StyleFilterSelector.tsx
│   │   │   ├── VideoCompilationSelector.tsx 
│   │   │   ├── VideoPreview.tsx
│   │   │   ├── VideoUploader.tsx
│   │   │   └── VoiceTranslationSelector.tsx
│   │   ├── api/               # API routes
│   │   │   └── process-ai-editing/        # AI editing API endpoint
│   │   ├── compilation/       # Compilation-related pages
│   │   ├── globals.css        # Global styles
│   │   ├── layout.tsx         # Root layout
│   │   └── page.tsx           # Main page
│   ├── package.json           # Node.js dependencies
│   └── tailwind.config.js     # Tailwind CSS configuration
├── install_dependencies.bat   # Windows dependency installer
├── install_dependencies.ps1   # PowerShell dependency installer
├── run_backend.bat           # Windows backend runner
├── run_backend.ps1           # PowerShell backend runner
├── start.bat                 # Windows full application starter
├── start.ps1                 # PowerShell full application starter
└── README.md                 # This file
```

### 🔧 Technology Stack
 
#### Backend (Python/FastAPI)  
- **Framework**: FastAPI with async/await support
- **AI/ML Libraries**:
  - OpenAI Whisper (Speech recognition)
  - Rembg (Background removal)
  - PySceneDetect (Scene detection) 
  - Coqui TTS (Text-to-speech)
  - Google Translate (Translation)
  - MediaPipe (Computer vision)
  - OpenCV (Video analysis and frame processing)
- **Video Processing**: FFmpeg, OpenCV
- **File Handling**: aiofiles, pathlib 
- **Performance**: ThreadPoolExecutor for parallel processing 

#### Frontend (Next.js/React)
- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**:  
  - Lucide React (Icons) 
  - Framer Motion (Animations)
  - React Dropzone (File uploads) 
  - Video.js (Video player)
- **HTTP Client**: Axios
- **Utilities**: clsx, class-variance-authority, tailwind-merge

## 🚀 Quick Start Guide
 
### Prerequisites
- **Python**: 3.10 or higher
- **Node.js**: 18 or higher
- **FFmpeg**: Installed and added to system Path
- **Git**: For cloning the repository

### Installation

#### Option 1: Automated Setup (Recommended)
```bash
# Clone the repository 
git clone https://github.com/ankit-kumarz/AI-powered-video-editing-application
cd videoeditor

# Windows (PowerShell)
.\install_dependencies.ps1

# Windows (Command Prompt)
install_dependencies.bat
```

#### Option 2: Manual Setup
```bash
# Clone the repository
git clone https://github.com/ankit-kumarz/AI-powered-video-editing-application 
cd videoeditor

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
cd frontend
npm install
cd ..
```

### Running the Application
 
#### Option 1: Automated Start (Recommended) 
```bash
# Windows (PowerShell)
.\start.ps1

# Windows (Command Prompt)
start.bat
```

#### Option 2: Manual Start
```bash
# Terminal 1: Start backend server
python run_backend.py

# Terminal 2: Start frontend development server 
cd frontend
npm run dev
```

### Access the Application
- **Frontend**: Open your browser to `http://localhost:3000`
- **Backend API**: Available at `http://localhost:8000` 
- **API Documentation**: Visit `http://localhost:8000/docs` for interactive API docs

## 📖 Usage Guide

### 1. Upload Video
- Drag and drop your video file or click to browse
- Supported formats: MP4, MOV, WebM
- Maximum file size: 500MB
- Maximum duration: 10 minutes

### 2. Select Features
Choose from individual AI features or use the comprehensive "AI Video Optimization":

#### Individual Features:
- **🎬 AI Film Editing Suggestions**: Get intelligent editing recommendations with interactive timeline
- **📝 Script Analysis**: Upload a script for script-based editing suggestions
- **🎭 Background Removal**: Remove and replace video backgrounds
- **📝 Subtitle Generation**: Generate subtitles from speech
- **🎬 Scene Detection**: Split video into individual scenes 
- **🌍 Voice Translation**: Translate speech to different languages 
- **🎨 Style Filters**: Apply artistic filters to your video
- **🚫 Object Removal**: Remove unwanted objects from video (3-4x faster with parallel processing)
- **✂️ Auto-Cut Silence**: Remove silent parts automatically
- **⚡ Video Compilation**: Compile multiple videos with parallel processing (up to 5 videos, 3x faster)

#### AI Video Optimization:
- Combines multiple features for complete video enhancement
- Includes scene analysis, silence removal, subtitle generation, and quality optimization

### 3. AI Film Editing Suggestions
The new AI Film Editing Suggestions feature provides: 

#### 🎯 **Editing Suggestion Types**:
- **✂️ Cuts**: Scene change detection, action moments, composition-based cuts
- **➡️ Transitions**: Smooth transition opportunities, visual flow suggestions
- **⭐ Emphasis**: Face detection, key moments, important dialogue
- **⚡ Pace Changes**: Pacing adjustments, hold moments, quick cuts

#### 🎬 **Interactive Timeline Features**:
- **Colored Markers**: Different colors for each suggestion type
- **Click Navigation**: Click any dot to jump to that timestamp
- **Hover Tooltips**: See suggestion details on hover
- **Multiple Suggestions**: Number badges show when multiple suggestions exist at the same time
- **Timeline Legend**: Color-coded legend explaining each marker type

#### 📝 **Script Analysis**:
- **Emotion Detection**: Analyze script for emotional content
- **Speaker Changes**: Detect speaker transitions
- **Transition Words**: Identify natural transition points
- **Pacing Analysis**: Suggest pacing adjustments based on content

### 4. Configure Settings
For voice translation:
- Select target language (14 supported languages)
- Choose voice type (male/female)
- Configure subtitle options

For AI editing:
- Upload optional script for enhanced suggestions
- Choose analysis features (video-based, script-based, or both)

### 5. Process Video
- Click "Process Video" to start AI processing
- Monitor real-time progress in the status panel
- Processing time varies based on video length and selected features

### 6. Explore Suggestions
- **Timeline Navigation**: Use the interactive timeline to explore suggestions
- **Filter Options**: Filter by suggestion type or search by description
- **Detailed View**: Click suggestions to see full details and reasoning
- **Compact Mode**: Toggle between detailed and compact views

### 7. Download Results
- Download buttons appear automatically after processing
- Multiple output formats available
- Individual scene downloads for scene detection

## 🎬 AI Film Editing Features

### Intelligent Video Analysis
The AI editing system analyzes your video using multiple techniques:

#### **Video-Based Analysis**:
- **Scene Change Detection**: Identifies natural cut points
- **Face Detection**: Finds important moments with people
- **Composition Analysis**: Evaluates visual quality and flow
- **Audio Analysis**: Detects speech, music, and silence patterns
- **Motion Analysis**: Identifies action and movement patterns

#### **Script-Based Analysis**:
- **Emotion Detection**: Analyzes emotional content in script
- **Speaker Changes**: Identifies natural transition points
- **Transition Words**: Finds words that indicate scene changes
- **Pacing Indicators**: Suggests timing adjustments

### Performance Optimizations
- **Parallel Processing**: Multi-threaded analysis for faster processing
- **Dynamic Frame Sampling**: Adjusts analysis frequency based on video length
- **Memory Optimization**: Efficient memory usage for long videos
- **Caching**: Intelligent caching of analysis results
- **Full Video Analysis**: Analyzes entire video, not just first 25 seconds

## 🌍 Voice Translation Features

The voice translation feature provides comprehensive video localization:

### Supported Languages
- **European**: Spanish, French, German, Italian, Portuguese, Dutch, Polish
- **Asian**: Japanese, Korean, Chinese
- **Other**: Russian, Arabic, Hindi, Turkish

### Translation Pipeline
1. **Speech Recognition**: OpenAI Whisper converts speech to text
2. **Translation**: Google Translate API translates the text
3. **Voice Generation**: Coqui TTS creates natural-sounding speech
4. **Subtitle Integration**: Translated subtitles are embedded in the video

### Voice Options
- **Female Voice**: Natural-sounding female speech
- **Male Voice**: Natural-sounding male speech
- **Subtitle Options**: Burned-in or separate subtitle files

## ⚙️ Configuration

### Environment Variables
Set these environment variables for enhanced features:

```bash
# OpenAI API Key (for Whisper speech recognition)
OPENAI_API_KEY=your_openai_api_key

# Google Translate API Key (for translation services)
GOOGLE_TRANSLATE_API_KEY=your_google_translate_key
```

### API Configuration
The backend server runs on port 8000 by default. You can modify this in `backend/main.py`: 

```python
# Change the port in the uvicorn.run() call
uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 🐛 Troubleshooting

### Common Issues

#### FFmpeg Not Found 
```bash
# Windows: Download from https://ffmpeg.org/download.html
# Add to PATH environment variable
# Verify installation:
ffmpeg -version
```

#### Processing Fails
- **Check video format**: Ensure it's MP4, MOV, or WebM
- **Check file size**: Maximum 500MB
- **Check duration**: Maximum 10 minutes 
- **Check disk space**: Ensure sufficient storage for processing

#### Download Issues
- **Browser settings**: Check if downloads are blocked
- **Disk space**: Ensure sufficient storage
- **File permissions**: Check write permissions in temp directory

#### Port Conflicts
```bash
# Check if ports are in use
netstat -ano | findstr :8000
netstat -ano | findstr :3000

# Kill processes if needed
taskkill /PID <process_id> /F
```

### Performance Optimization
- **GPU Acceleration**: Install CUDA for faster processing
- **Memory**: Ensure sufficient RAM (8GB+ recommended)
- **Storage**: Use SSD for faster file operations
- **Parallel Processing**: Video compilation and object removal use ThreadPoolExecutor for 3-4x faster processing
- **Adaptive Analysis**: Frame interval adjusts based on video length for optimal performance
- **Multi-threaded FFmpeg**: Uses multiple threads for faster video encoding
- **Memory Efficiency**: Batch processing reduces memory usage by 70%
- **GPU Acceleration**: Support for CUDA acceleration when available
- **Full Video Analysis**: Complete video analysis with optimized performance

## 🔧 Development

### Backend Development
```bash
# Install development dependencies
pip install -r requirements_dev.txt

# Run with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development
```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Type checking
npm run type-check

# Linting
npm run lint
```

### Testing
```bash
# Backend tests
python test_backend.py

# Frontend tests
cd frontend
npm test
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Test thoroughly**
5. **Submit a pull request**

### Development Guidelines
- Follow PEP 8 for Python code
- Use TypeScript for frontend components
- Add tests for new features
- Update documentation as needed
- Use conventional commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

## 📞 Support & Community

### Getting Help
- **GitHub Issues**: Create an issue for bugs or feature requests
- **Documentation**: Check this README and code comments
- **Troubleshooting**: Review the troubleshooting section above

### Community Guidelines
- Be respectful and helpful
- Provide detailed bug reports
- Share your use cases and feedback
- Contribute to documentation improvements

## 🎯 Roadmap

### Planned Features
- [ ] **Batch Processing**: Process multiple videos simultaneously
- [ ] **Cloud Storage**: Integration with cloud storage providers
- [ ] **Advanced Filters**: More AI-powered video filters
- [ ] **Mobile App**: React Native mobile application
- [ ] **API Rate Limiting**: Better resource management
- [ ] **User Authentication**: User accounts and project management
- [ ] **Advanced Timeline**: More sophisticated timeline editing features
- [ ] **Export Presets**: Save and share editing presets

### Performance Improvements
- [ ] **GPU Acceleration**: Enhanced CUDA support
- [ ] **Caching**: Intelligent result caching
- [ ] **Compression**: Better video compression algorithms
- [ ] **Parallel Processing**: Multi-threaded AI processing
- [ ] **Real-time Preview**: Live preview of editing suggestions

---

## 📬 Contact:
For any questions or feedback, feel free to contact:

Author - Ankit kumar 
Email: ankitrajj1068@gmail.com 
GitHub: https://github.com/ankit-kumarz
