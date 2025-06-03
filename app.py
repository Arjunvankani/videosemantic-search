import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import io
from sentence_transformers import SentenceTransformer
import tempfile
import hashlib
import os

@st.cache_resource
def get_model():
    return SentenceTransformer('clip-ViT-B-32')

def get_video_hash(video_bytes):
    """Generate hash from video bytes instead of file path"""
    return hashlib.md5(video_bytes).hexdigest()

@st.cache_data(show_spinner=False)
def extract_frames(_video_bytes, video_hash):
    """Extract frames using video bytes and hash for caching"""
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(_video_bytes)
        video_path = tmp_file.name
    
    try:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to extract 2 frames per second
        frame_indices = np.arange(0, total_frames, fps/2, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize to 224x224 as required by CLIP
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
        
        cap.release()
        return frames
    finally:
        # Clean up temporary file
        if os.path.exists(video_path):
            os.unlink(video_path)

@st.cache_data(show_spinner=False)
def get_frame_embeddings(_frames, video_hash):
    """Generate embeddings with proper caching"""
    model = get_model()
    # Convert frames to PIL Images
    pil_frames = [Image.fromarray(frame) for frame in _frames]
    # Get embeddings
    embeddings = model.encode(pil_frames)
    return embeddings

def search_frames(query_text, frames, embeddings):
    model = get_model()
    # Get text embedding
    text_embedding = model.encode([query_text])[0]
    # Calculate similarities
    similarities = [np.dot(text_embedding, frame_emb) / 
                   (np.linalg.norm(text_embedding) * np.linalg.norm(frame_emb))
                   for frame_emb in embeddings]
    # Sort frames by similarity
    sorted_pairs = sorted(zip(similarities, frames), key=lambda x: x[0], reverse=True)
    return sorted_pairs

def main():
    st.title('Video / Reel Context Search')
    st.write('Upload a video and search for specific moments using natural language!')
    
    # File uploader
    uploaded_file = st.file_uploader('Choose a video file', type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # Read video bytes once
        video_bytes = uploaded_file.read()
        video_hash = get_video_hash(video_bytes)
        
        # Store video info in session state to persist across reruns
        if 'current_video_hash' not in st.session_state or st.session_state.current_video_hash != video_hash:
            st.session_state.current_video_hash = video_hash
            st.session_state.processing_complete = False
        
        # Only process if not already done for this video
        if not st.session_state.get('processing_complete', False):
            # Extract frames
            with st.spinner('Extracting frames from video...'):
                frames = extract_frames(video_bytes, video_hash)
                st.session_state.frames = frames
                st.success(f'Extracted {len(frames)} frames!')
            
            # Get embeddings for frames
            with st.spinner('Generating embeddings...'):
                embeddings = get_frame_embeddings(frames, video_hash)
                st.session_state.embeddings = embeddings
                st.success('Generated embeddings!')
                st.session_state.processing_complete = True
        else:
            # Use cached data from session state
            frames = st.session_state.frames
            embeddings = st.session_state.embeddings
            st.success(f'Using cached data - {len(frames)} frames ready for search!')
        
        # Search interface
        query = st.text_input('Enter your search query:', key='search_query')
        if query:
            with st.spinner('Searching...'):
                results = search_frames(query, frames, embeddings)
            
            # Display results
            st.subheader('Search Results')
            cols = st.columns(3)
            for idx, (similarity, frame) in enumerate(results[:6]):
                with cols[idx % 3]:
                    st.image(frame, caption=f'Similarity: {similarity:.2f}')

if __name__ == '__main__':
    main()