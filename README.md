# Video Semantic Search

This Streamlit application allows users to upload videos and perform semantic search on video frames using natural language queries. The application uses OpenAI's CLIP model to generate embeddings for video frames and find relevant moments based on text descriptions.

## Features

- Video upload support (MP4, AVI, MOV)
- Automatic frame extraction
- Semantic search using natural language
- Real-time results with similarity scores
- Visual display of matching frames

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd videosemantic-search
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Upload a video file using the file uploader

4. Wait for the frames to be extracted and embeddings to be generated

5. Enter a text query to search for specific moments in the video

6. View the results showing the most relevant frames matching your query

## How it Works

The application uses the following process:
1. Extracts frames from the uploaded video at regular intervals
2. Generates embeddings for each frame using the CLIP model
3. Converts user's text query into an embedding
4. Calculates similarity between the query and frame embeddings
5. Displays the most similar frames

## Requirements

- Python 3.7+
- CUDA-compatible GPU (optional, for faster processing)
- Sufficient RAM for processing video frames

