import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import os
from datetime import datetime
import base64
from werkzeug.utils import secure_filename
from PIL import Image
import tempfile
import time

# Set page config for a wider layout
st.set_page_config(layout="wide", page_title="Face Verification System")

# Custom CSS for better UI
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        height: 3em;
        margin: 1em 0;
    }
    .verification-results {
        padding: 1em;
        border-radius: 0.5em;
        margin: 1em 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'camera' not in st.session_state:
    st.session_state.camera = cv2.VideoCapture(0)
if 'verification_history' not in st.session_state:
    st.session_state.verification_history = []

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def get_dominant_gender(gender_dict):
    """Extract dominant gender from DeepFace gender dictionary"""
    return max(gender_dict.items(), key=lambda x: x[1])[0]

def format_gender_info(gender_dict):
    """Format gender probabilities into a readable string"""
    dominant = get_dominant_gender(gender_dict)
    probability = gender_dict[dominant]
    return f"{dominant} ({probability:.1f}%)"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def capture_frame():
    """Capture a frame from the webcam"""
    success, frame = st.session_state.camera.read()
    if success:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

def analyze_face(image_path):
    """Analyze face to get demographic information"""
    try:
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['age', 'gender', 'race', 'emotion'],
            enforce_detection=False
        )
        result = result[0] if isinstance(result, list) else result
        
        # Process the gender result
        if isinstance(result['gender'], dict):
            result['gender_dict'] = result['gender']
            result['gender'] = get_dominant_gender(result['gender'])
        
        return result
    except Exception as e:
        st.error(f"Error during face analysis: {str(e)}")
        return None

def verify_faces(uploaded_image_path, current_frame):
    """Verify faces between uploaded image and current frame"""
    try:
        # Save current frame temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_frame_path = temp_file.name
            current_frame_bgr = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(temp_frame_path, current_frame_bgr)

        # Verify faces
        result = DeepFace.verify(
            img1_path=uploaded_image_path,
            img2_path=temp_frame_path,
            enforce_detection=False
        )
        
        # Get additional face analysis
        analysis = analyze_face(temp_frame_path)

        # Clean up temporary file
        os.remove(temp_frame_path)

        if analysis:
            result.update({
                'age': analysis['age'],
                'gender': analysis['gender'],
                'gender_dict': analysis.get('gender_dict', {}),
                'dominant_race': analysis['dominant_race'],
                'dominant_emotion': analysis['dominant_emotion']
            })

        return result
    except Exception as e:
        st.error(f"Error during face verification: {str(e)}")
        return None

def main():
    st.title("🎯 Advanced Face Verification System")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📤 Upload Reference Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Reference Image", use_container_width=True)
            
            # Save uploaded file
            filename = secure_filename(uploaded_file.name)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            with open(filepath, 'wb') as f:
                f.write(uploaded_file.getvalue())
    
    if uploaded_file is not None:
        # Main content area - split into two columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📹 Live Camera Feed")
            # Placeholder for webcam feed
            frame_placeholder = st.empty()
            
            # Verification results placeholder
            results_placeholder = st.empty()
            
            # Capture and verify button
            if st.button("🔍 Capture and Verify", key="verify_button"):
                current_frame = capture_frame()
                if current_frame is not None:
                    # Display captured frame
                    frame_placeholder.image(current_frame, channels="RGB", use_container_width=True)
                    
                    # Perform verification
                    verification_result = verify_faces(filepath, current_frame)
                    
                    if verification_result:
                        # Add timestamp to result
                        verification_result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        verification_result['reference_name'] = os.path.splitext(filename)[0]
                        
                        # Add to history
                        st.session_state.verification_history.insert(0, verification_result)
                        
                        # Display results
                        results_container = results_placeholder.container()
                        with results_container:
                            st.markdown("### 📊 Verification Results")
                            cols = st.columns(2)
                            with cols[0]:
                                st.metric("Match Status", "✅ Verified" if verification_result['verified'] else "❌ Not Verified")
                                st.metric("Reference Name", verification_result['reference_name'])
                                st.metric("Similarity Distance", f"{verification_result['distance']:.4f}")
                            with cols[1]:
                                st.metric("Estimated Age", f"{verification_result['age']}")
                                # Display gender with probability
                                if 'gender_dict' in verification_result:
                                    gender_info = format_gender_info(verification_result['gender_dict'])
                                else:
                                    gender_info = verification_result['gender']
                                st.metric("Gender", gender_info)
                                st.metric("Dominant Race", verification_result['dominant_race'])
                                st.metric("Dominant Emotion", verification_result['dominant_emotion'])
            
            # Live preview
            while True:
                current_frame = capture_frame()
                if current_frame is not None:
                    frame_placeholder.image(current_frame, channels="RGB", use_container_width=True)
                time.sleep(0.1)
                
        with col2:
            st.header("📝 Verification History")
            for idx, result in enumerate(st.session_state.verification_history):
                with st.expander(f"Verification {idx + 1} - {result['timestamp']}", expanded=(idx == 0)):
                    st.write(f"**Status:** {'✅ Verified' if result['verified'] else '❌ Not Verified'}")
                    st.write(f"**Reference:** {result['reference_name']}")
                    st.write(f"**Distance:** {result['distance']:.4f}")
                    st.write(f"**Age:** {result['age']} years")
                    if 'gender_dict' in result:
                        st.write(f"**Gender:** {format_gender_info(result['gender_dict'])}")
                    else:
                        st.write(f"**Gender:** {result['gender']}")
                    st.write(f"**Race:** {result['dominant_race']}")
                    st.write(f"**Emotion:** {result['dominant_emotion']}")
            
    else:
        st.info("👆 Please upload a reference image in the sidebar to start face verification.")

if __name__ == '__main__':
    main()