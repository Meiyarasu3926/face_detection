import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
from deepface import DeepFace
import tempfile

# Set up directories
UPLOAD_DIR = "registered_faces"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Define the expected columns for each CSV file
REGISTERED_USERS_COLUMNS = [
    'employee_id', 'name', 'registered_gender', 'gender_confidence', 
    'image_path', 'registration_date'
]

ATTENDANCE_LOG_COLUMNS = [
    'employee_id', 'name', 'registered_gender', 'verified_gender',
    'gender_confidence', 'confidence', 'timestamp'
]

# Initialize Streamlit session state
if 'registered_users' not in st.session_state:
    st.session_state.registered_users = {}

def initialize_csv_files():
    """Initialize CSV files with correct headers if they don't exist"""
    if not os.path.exists('registered_users.csv'):
        pd.DataFrame(columns=REGISTERED_USERS_COLUMNS).to_csv('registered_users.csv', index=False)
    if not os.path.exists('attendance_log.csv'):
        pd.DataFrame(columns=ATTENDANCE_LOG_COLUMNS).to_csv('attendance_log.csv', index=False)

def load_registered_users():
    """Load registered users from CSV"""
    try:
        if os.path.exists('registered_users.csv'):
            df = pd.read_csv('registered_users.csv')
            if len(df) > 0:
                return dict(zip(df['employee_id'], df.to_dict('records')))
        return {}
    except:
        initialize_csv_files()
        return {}

def save_to_csv(data, filename):
    """Save data to CSV with proper column handling"""
    try:
        df_new = pd.DataFrame([data])
        if os.path.exists(filename):
            # Read existing data
            try:
                df_existing = pd.read_csv(filename)
            except:
                df_existing = pd.DataFrame(columns=df_new.columns)
            
            # Combine and save
            df_updated = pd.concat([df_existing, df_new], ignore_index=True)
            df_updated.to_csv(filename, index=False)
        else:
            df_new.to_csv(filename, index=False)
    except:
        if filename == 'registered_users.csv':
            pd.DataFrame(columns=REGISTERED_USERS_COLUMNS).to_csv(filename, index=False)
        else:
            pd.DataFrame(columns=ATTENDANCE_LOG_COLUMNS).to_csv(filename, index=False)

def get_dominant_gender(gender_dict):
    """Extract dominant gender from DeepFace analysis"""
    if isinstance(gender_dict, dict):
        dominant_gender = max(gender_dict.items(), key=lambda x: float(str(x[1])))[0]
        confidence = float(str(gender_dict[dominant_gender]))
        return dominant_gender, confidence
    return str(gender_dict), None

def analyze_face(image):
    """Analyze face for demographics"""
    try:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        analysis = DeepFace.analyze(
            img_path=image,
            actions=['gender'],
            enforce_detection=True,
            detector_backend='mtcnn'
        )
        result = analysis[0] if isinstance(analysis, list) else analysis
        gender, confidence = get_dominant_gender(result['gender'])
        
        return {
            'gender': gender,
            'gender_confidence': confidence
        }
    except Exception as e:
        st.error("No clear face detected. Please try again.")
        return None

def enhanced_spoof_detection(frame):
    """Enhanced spoof detection using multiple techniques"""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
    color_var = np.sum([cv2.Laplacian(frame[:,:,i], cv2.CV_64F).var() for i in range(3)])
    edges = cv2.Canny(gray_frame, 100, 200)
    edge_density = np.mean(edges)
    blur = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    moire = np.abs(gray_frame.astype(np.float32) - blur.astype(np.float32))
    moire_score = np.mean(moire)

    return (
        laplacian_var > 100 and
        color_var > 300 and
        edge_density > 10 and
        moire_score < 10
    )

def register_new_employee():
    """Register a new employee using only file upload"""
    st.title("🏢 Company Attendance System")
    st.header("👤 Register New Employee")

    employee_id = st.text_input("Enter Employee ID")
    employee_name = st.text_input("Enter Employee Name")
    
    if not employee_id or not employee_name:
        st.warning("Please enter both Employee ID and Name")
        return

    uploaded_file = st.file_uploader(
        "Upload Registration Photo (clear front-facing photo)", 
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        analysis = analyze_face(image)
        if analysis is None:
            return

        temp_path = os.path.join(UPLOAD_DIR, f"{employee_id}.jpg")
        cv2.imwrite(temp_path, image)

        employee_data = {
            'employee_id': employee_id,
            'name': employee_name,
            'registered_gender': analysis['gender'],
            'gender_confidence': analysis.get('gender_confidence', 100),
            'image_path': temp_path,
            'registration_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        st.session_state.registered_users[employee_id] = employee_data
        save_to_csv(employee_data, 'registered_users.csv')

        st.success(f"Employee {employee_name} registered successfully!")
        st.write("### Detected Information:")
        st.write(f"- Gender: {analysis['gender']} ({analysis.get('gender_confidence', 100):.2f}% confidence)")
        st.image(uploaded_file, caption=f"Registration Photo - {employee_name}")

def verify_employee():
    """Verify employee attendance using live camera with anti-spoofing"""
    st.title("🏢 Company Attendance System")
    st.header("📸 Live Verification")

    employee_id = st.text_input("Enter Employee ID")

    if not employee_id:
        st.warning("Please enter an Employee ID")
        return

    if employee_id not in st.session_state.registered_users:
        st.error("Employee ID not found. Please register first.")
        return

    employee_data = st.session_state.registered_users[employee_id]
    st.write(f"Verifying identity for: {employee_data['name']}")

    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    verification_button = st.button("Verify Identity")
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                st.error("Camera not working. Please check your device.")
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame, channels="RGB")

            if verification_button:
                if not enhanced_spoof_detection(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                    st.error("Spoof detected! Please use a real face, not a photo or screen.")
                    break

                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, "capture.jpg")
                cv2.imwrite(temp_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                try:
                    result = DeepFace.verify(
                        img1_path=employee_data['image_path'],
                        img2_path=temp_path,
                        enforce_detection=True,
                        detector_backend='mtcnn'
                    )

                    live_analysis = analyze_face(frame)

                    if result['verified']:
                        st.success("Identity Verified Successfully! ✅")
                        
                        attendance_data = {
                            'employee_id': employee_id,
                            'name': employee_data['name'],
                            'registered_gender': employee_data['registered_gender'],
                            'verified_gender': live_analysis['gender'] if live_analysis else 'Unknown',
                            'gender_confidence': live_analysis.get('gender_confidence', 0) if live_analysis else 0,
                            'confidence': (1 - result['distance']) * 100,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        save_to_csv(attendance_data, 'attendance_log.csv')
                        
                        st.write("### Verification Details")
                        st.write(f"- Name: {employee_data['name']}")
                        st.write(f"- Registered Gender: {employee_data['registered_gender']}")
                        if live_analysis:
                            st.write(f"- Verified Gender: {live_analysis['gender']} "
                                   f"({live_analysis.get('gender_confidence', 100):.2f}% confidence)")
                        st.write(f"- Verification Confidence: {attendance_data['confidence']:.2f}%")
                        st.write(f"- Timestamp: {attendance_data['timestamp']}")
                    else:
                        st.error("Verification Failed! Identity mismatch. ❌")
                except Exception as e:
                    st.error("Verification failed. Please try again.")
                finally:
                    os.remove(temp_path)
                break
    finally:
        camera.release()

def view_attendance():
    """View attendance records"""
    st.title("🏢 Company Attendance System")
    st.header("📊 Attendance Records")

    if os.path.exists('attendance_log.csv'):
        try:
            df = pd.read_csv('attendance_log.csv')
            
            st.subheader("Filters")
            col1, col2 = st.columns(2)
            
            with col1:
                selected_date = st.date_input("Select Date", datetime.now())
            
            with col2:
                gender_column = 'registered_gender'
                if gender_column in df.columns and not df[gender_column].empty:
                    selected_gender = st.selectbox(
                        "Select Gender",
                        ['All'] + df[gender_column].dropna().unique().tolist()
                    )
                else:
                    selected_gender = 'All'
            
            filtered_df = df.copy()
            
            if 'timestamp' in filtered_df.columns:
                filtered_df['date'] = pd.to_datetime(filtered_df['timestamp']).dt.date
                filtered_df = filtered_df[filtered_df['date'] == selected_date]
            
            if selected_gender != 'All' and gender_column in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[gender_column] == selected_gender]
            
            st.dataframe(filtered_df)

            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered Attendance Log",
                data=csv,
                file_name=f"attendance_log_{selected_date}.csv",
                mime="text/csv"
            )
            
            st.subheader("Daily Statistics")
            st.write(f"Total Attendance: {len(filtered_df)}")
            
            if gender_column in filtered_df.columns and not filtered_df[gender_column].empty:
                st.write("Gender Distribution:")
                st.write(filtered_df[gender_column].value_counts())
        except:
            initialize_csv_files()
            st.info("No attendance records found.")
    else:
        initialize_csv_files()
        st.info("No attendance records found.")

def main():
    """Main app logic"""
    st.set_page_config(page_title="Company Attendance System", page_icon="🏢")
    
    # Initialize CSV files if they don't exist
    initialize_csv_files()
    
    if 'registered_users' not in st.session_state or not st.session_state.registered_users:
        st.session_state.registered_users = load_registered_users()

    menu = st.sidebar.radio(
        "Navigation",
        ["Register Employee", "Verify Attendance", "View Attendance Log"]
    )

    if menu == "Register Employee":
        register_new_employee()
    elif menu == "Verify Attendance":
        verify_employee()
    elif menu == "View Attendance Log":
        view_attendance()

if __name__ == "__main__":
    main()
