from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from deepface import DeepFace
import os
from datetime import datetime
import base64
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize camera
camera = cv2.VideoCapture(0)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def get_frame():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename,
            'name': os.path.splitext(filename)[0]  # Extract name from filename
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/verify', methods=['POST'])
def verify():
    try:
        # Capture current frame from webcam
        success, frame = camera.read()
        if not success:
            return jsonify({'error': 'Failed to capture webcam frame'}), 500
        
        # Save current frame temporarily
        temp_frame_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_frame.jpg')
        cv2.imwrite(temp_frame_path, frame)
        
        # Get uploaded image path
        uploaded_image = request.form.get('uploaded_image')
        if not uploaded_image:
            return jsonify({'error': 'No reference image provided'}), 400
        
        uploaded_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image)
        
        # Verify faces
        result = DeepFace.verify(
            img1_path=uploaded_path,
            img2_path=temp_frame_path,
            enforce_detection=False
        )
        
        # Clean up temporary file
        os.remove(temp_frame_path)
        
        name = os.path.splitext(uploaded_image)[0]
        return jsonify({
            'verified': result['verified'],
            'distance': result['distance'],
            'threshold': result['threshold'],
            'name': name
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)