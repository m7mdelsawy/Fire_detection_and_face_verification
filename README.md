🔥 Fire Detection & 👤 Face Verification System
This project is a computer vision–based AI system that combines Fire/Smoke Detection and Face Verification into a single deployable application. It is designed for security services, smart compounds, and city surveillance scenarios.

The system can:

Detect fire or smoke in images / video frames
Verify whether a detected face belongs to a known person or classify it as unknown
🚀 Features
🔥 Fire & Smoke Detection
Deep Learning–based fire/smoke classifier
Trained model stored as .pth
Optimized for real-time inference
👤 Face Verification (Closed-set + Unknown)
Uses face embeddings for verification

Supports:

Known identities
Automatic unknown person detection
Embeddings and metadata stored for fast lookup

🧠 Model Assets
Pretrained PyTorch models
Saved embeddings and metadata using Pickle
🌐 Deployment Ready
Can be deployed using Streamlit / FastAPI / Hugging Face Spaces
Lightweight and modular design
📁 Project Structure
Fire_detection_and_face_verification/
│
├── notebooks/
│   └── fire-and-smoke-detection-training.ipynb
│
├── app.py                          # Main application entry point
├── best_mobile_accuracy_model.pth  # Fire/Smoke detection model
├── face_recognition_model.pth      # Face embedding model
├── face_embeddings.pkl             # Stored face embeddings
├── face_metadata.pkl               # Names / IDs mapping
├── requirements_txt.txt            # Project dependencies
├── readme_huggingface.md            # Hugging Face specific README
└── README.md                        # Project documentation
⚙️ Installation
Clone the repository:

git clone https://github.com/your-username/Fire_detection_and_face_verification.git
cd Fire_detection_and_face_verification
Install dependencies:

pip install -r requirements_txt.txt
💡 Tip: It is recommended to use a virtual environment.

▶️ Usage
Run the application:

python app.py
The app will:

Load the fire detection model
Load face recognition model + embeddings
Perform inference on input images / frames
🧠 How It Works
Fire Detection Pipeline
Input image/frame
CNN-based classifier
Output: Fire / Smoke / Normal
Face Verification Pipeline
Face detection & alignment

Feature extraction (embedding vector)

Similarity comparison with stored embeddings

Decision:

Known person
Unknown
📊 Models & Files
File	Description
best_mobile_accuracy_model.pth	Fire & smoke detection model
face_recognition_model.pth	Face embedding model
face_embeddings.pkl	Encoded face vectors
face_metadata.pkl	Identity metadata
🧪 Training
Training notebooks are available in the notebooks/ directory.

You can:

Retrain fire/smoke detection
Update face embeddings with new identities
🛡️ Use Cases
Smart compounds
City surveillance systems
Industrial safety monitoring
Private security services
🔮 Future Improvements
Real-time video stream support
Multi-camera tracking
Alert system (SMS / Email / Dashboard)
Model optimization for edge devices
👨‍💻 Author
Mohamed Elsawy AI Engineering Student – Mansoura University

📜 License
This project is for educational and research purposes. Licensing can be added if required.

⭐ If you like this project, feel free to star the repository!
