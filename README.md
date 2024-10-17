# Face Recognition Attendance System

This project implements a face recognition system for attendance tracking using Python, OpenCV, TensorFlow, and scikit-learn.

## Important Files

- `sample.py`: Main Python script containing the FaceTrainer class and program logic
- `requirements.txt`: List of required Python packages
- `face_recognition.db`: SQLite database for storing user and training session information
- `data/`: Directory containing captured face images for training

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/face-recognition-attendance-system.git
   cd face-recognition-attendance-system
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   # On Linux or MacOS:   source venv/bin/activate  
   # On Windows, use:     venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script: 


Follow the on-screen menu to:
1. Generate datasets for multiple individuals
2. Train the classifier
3. Perform face recognition

## System Overview

This Face Recognition Attendance System uses a combination of traditional computer vision techniques and machine learning algorithms to detect and recognize faces. Here's a brief overview of the key components and their purposes:

1. **Face Detection**: 
   - Algorithm: Haar Cascade Classifier
   - Purpose: Rapidly detect faces in input images or video streams
   - Benefits: Fast, efficient, and works well for frontal faces

2. **Feature Extraction**:
   - Algorithms: SIFT (Scale-Invariant Feature Transform) and MobileNetV2
   - Purpose: Extract distinctive features from detected faces
   - Benefits: 
     - SIFT: Robust to scale, rotation, and illumination changes
     - MobileNetV2: Provides deep learning-based features, capturing high-level facial characteristics

3. **Face Recognition**:
   - Algorithm: Support Vector Machine (SVM)
   - Purpose: Classify extracted features to identify individuals
   - Benefits: Effective for multi-class classification, works well with high-dimensional data

4. **Data Preprocessing**:
   - Techniques: Grayscale conversion, histogram equalization
   - Purpose: Enhance image quality and normalize input data
   - Benefits: Improves feature extraction and recognition accuracy

5. **Database Management**:
   - Technology: SQLite
   - Purpose: Store user information and training session statistics
   - Benefits: Lightweight, serverless, easy to integrate with Python

## Performance and Statistics

The system's performance can vary depending on the quality and quantity of training data. Typically, you can expect:

- Face Detection Accuracy: ~95-99% (under good lighting conditions)
- Face Recognition Accuracy: ~85-95% (with a well-trained model)
- Processing Speed: 5-10 frames per second (on average hardware)

To view detailed statistics for your trained model, check the output after the training process or query the `training_sessions` table in the SQLite database.

## Future Improvements

- Implement more advanced face detection algorithms (e.g., MTCNN or RetinaFace)
- Explore deep learning-based face recognition models (e.g., FaceNet or DeepFace)
- Add real-time attendance logging and reporting features
- Implement data augmentation techniques to improve model robustness


## License

## Technical Details

### Face Detection
- **Algorithm**: Haar Cascade Classifier
- **Implementation**: OpenCV's `cv2.CascadeClassifier`
- **Model**: `haarcascade_frontalface_default.xml`
- **Performance**: Fast but may struggle with non-frontal faces or poor lighting conditions
- **Future Improvement**: Consider using MTCNN or RetinaFace for more robust face detection

### Feature Extraction
- **Deep Learning Model**: MobileNetV2
- **Framework**: TensorFlow / Keras
- **Input Shape**: 224x224x3 (RGB image)
- **Output**: 1280-dimensional feature vector
- **Pre-training**: ImageNet weights
- **Usage**: Transfer learning, using the model as a feature extractor
- **Advantages**: Lightweight, efficient, and provides rich feature representations

### Face Recognition
- **Algorithm**: Support Vector Machine (SVM)
- **Implementation**: scikit-learn's `SVC` (Support Vector Classification)
- **Kernel**: RBF (Radial Basis Function)
- **Probability Estimates**: Enabled for confidence scoring
- **Training**: Performed on the combined features from MobileNetV2

### Data Preprocessing
- **Color Conversion**: BGR to RGB for MobileNetV2 input
- **Resizing**: Images resized to 224x224 for consistency
- **Normalization**: Pixel values scaled to [-1, 1] range for MobileNetV2

### Performance Metrics
- **Accuracy**: Overall correct predictions / total predictions
- **Precision**: True positives / (true positives + false positives)
- **Recall**: True positives / (true positives + false negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions per class

### Database Management
- **Database**: SQLite3
- **Tables**: 
  - `users`: Stores user information
  - `training_sessions`: Records details of each training session
  - `recognition_sessions`: Logs performance metrics for recognition sessions

### Optimization Techniques
- **Frame Skipping**: Processing every 3rd frame during real-time recognition
- **Confidence Thresholding**: Recognitions below 50% confidence are labeled as "Unknown"
- **Histogram Equalization**: Applied to improve contrast in face images

<!-- more to go here -->

### Future Enhancements


### System Requirements
- **Python**: 3.7+
- **Key Libraries**: 
  - OpenCV 4.5+
  - TensorFlow 2.5+
  - scikit-learn 0.24+
  - NumPy 1.19+

This Face Recognition Attendance System combines traditional computer vision techniques with modern deep learning approaches to create an efficient and accurate solution. The use of transfer learning with MobileNetV2 allows for rich feature extraction without the need for extensive training data, while the SVM classifier provides a robust and interpretable classification mechanism. The system is designed to be scalable and can be further optimized for specific deployment scenarios.
