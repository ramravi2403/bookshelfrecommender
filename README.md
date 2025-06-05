
3. Launch the dashboard:
```bash
python recommender/gradio_dashboard.py
```

## 📚 Project Structure

### Core Components

#### 1. Recommender System (`recommender/`)
- `semantic_book_recommender.py`: Core recommendation engine using sentence transformers
  - Implements semantic search for books
  - Handles emotion and genre classification
  - Processes natural language queries with filters
- `gradio_dashboard.py`: Web interface for the system
  - Upload and process bookshelf images
  - Search and filter books
  - Display results with book covers and details

#### 2. Object Detection (`object_detection/`)
- `YoloModel.py`: YOLOv8 implementation for book spine detection
- `ResultsProcessor.py`: Processes detection results and extracts book spines
- Features:
  - Book spine detection
  - Image cropping and preprocessing
  - Result visualization

#### 3. OCR Processing (`ocr/`)
- `OCRModels/`: Contains OCR implementations
  - `GoogleVisionOCR.py`: Google Cloud Vision API integration
- `Evaluator/`: OCR evaluation tools
- `Metrics/`: Performance measurement tools
- Features:
  - Text extraction from book spines
  - OCR accuracy evaluation
  - Performance metrics calculation

#### 4. OCR Parser (`ocr_parser/`)
- `BookOCRProcessor.py`: Processes OCR results
- `language_model_adapter.py`: LangChain integration for LLMs
- `ModelFactory.py`: Factory for different LLM providers
- `book_processing_service.py`: Book metadata processing
- Features:
  - Text cleaning and normalization
  - Book metadata extraction
  - Integration with multiple LLM providers (GPT, Claude, Gemini)

### Data Flow

1. **Image Processing Pipeline**:
   - Upload bookshelf image through Gradio interface
   - YOLO model detects book spines
   - Google Vision OCR extracts text
   - LangChain processes text with LLMs
   - Results stored in processed dataset

2. **Recommendation Pipeline**:
   - User enters search query
   - Query processed for filters and intent
   - Semantic search finds matching books
   - Results filtered by genre, emotion, rating
   - Displayed in interactive gallery

## 🔧 Configuration

### Model Selection
The system supports multiple LLM providers:
- OpenAI GPT models
- Anthropic Claude models
- Google Gemini models

Configure the desired model in `ocr_parser/ModelFactory.py`

### OCR Settings
- Google Cloud Vision API credentials required
- Configure in `ocr/Credentials.py`

## �� Features

### Book Detection
- Automatic book spine detection
- Multiple book handling
- Robust to different shelf arrangements

### Text Processing
- OCR text extraction
- LLM-powered text cleaning
- Metadata extraction

### Recommendation Engine
- Semantic search
- Emotion-based filtering
- Genre classification
- Rating-based filtering
- Author-based search

### User Interface
- Interactive web dashboard
- Real-time processing
- Visual result display
- Multiple filter options

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

[Add your license information here]

## �� Acknowledgments

- YOLOv8 for object detection
- Google Cloud Vision for OCR
- LangChain for LLM integration
- Gradio for the web interface