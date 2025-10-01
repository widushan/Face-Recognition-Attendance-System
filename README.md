# 🎓 Face Recognition Attendance System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.3-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11.0-orange.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.21-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**An intelligent attendance management system powered by AI and computer vision**

[![GitHub stars](https://img.shields.io/github/stars/yourusername/face-recognition-attendance-system.svg?style=social&label=Star)](https://github.com/yourusername/face-recognition-attendance-system)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/face-recognition-attendance-system.svg?style=social&label=Fork)](https://github.com/yourusername/face-recognition-attendance-system)

</div>

---

## 📸 Screenshots

<div align="center">

### 🏠 Dashboard
<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/0eb89bfd-c8bb-46ab-8f87-f1e3fc890da3" />

### 👤 Add Student
<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/75bc017c-b5d9-43bb-8e07-46122293618f" />

### 📝 Mark Attendance
<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/6242e71d-8334-481b-9bf8-a7d8632a1f2b" />

### 📊 Attendance Records
<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/f18da0ef-db7d-4ca0-a0d6-1612604afaf7" />

</div>

---

## 🚀 Features

### ✨ Core Functionality
- **🎯 Real-time Face Recognition** - Live camera feed with instant student identification
- **📚 Module-based Attendance** - Separate tracking for different subjects/courses
- **🔄 Duplicate Prevention** - Smart system prevents multiple attendance for same module/day
- **📊 Comprehensive Records** - Detailed attendance history with filtering options
- **📁 CSV Export** - Download attendance data by module or date range

### 🛠️ Technical Features
- **🤖 AI-Powered Recognition** - MediaPipe + OpenCV + Scikit-learn pipeline
- **💾 SQLite Database** - Lightweight, reliable data storage
- **🎨 Modern UI** - Bootstrap 5 responsive design
- **📱 Cross-platform** - Works on Windows, macOS, and Linux
- **🔒 Secure & Private** - All data stored locally

### 📋 Module Management
- **Module 1, Module 2, Module 3, Module 4** - Predefined course modules
- **Batch Selection** - Batch 1-5 support for different student groups
- **Degree Programs** - Engineering, IT, Accounting, Business, Arts

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11 or higher
- Webcam/Camera access
- Git

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/face-recognition-attendance-system.git
   cd face-recognition-attendance-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the system**
   - Open your browser and go to `http://localhost:5000`
   - Allow camera permissions when prompted

---

## 📖 Usage Guide

### 1. Adding Students
- Navigate to "Add Student" page
- Fill in student details (Name, Registration No, Batch, Degree)
- Capture 50 face images for training
- Train the model using the "Train Model" button

### 2. Marking Attendance
- Go to "Mark Attendance" page
- Select the appropriate Module Code
- Click "Start" to begin live recognition
- Students are automatically recognized and marked present
- System prevents duplicate attendance for same module/day

### 3. Viewing Records
- Access "Attendance Records" page
- Filter by time period (Daily, Weekly, Monthly)
- Filter by specific Module Code
- Download CSV reports for individual modules

---

## 🏗️ Project Structure

```
face-recognition-attendance-system/
├── 📁 static/
│   ├── 📁 css/          # Stylesheets
│   ├── 📁 js/           # JavaScript files
│   └── 📁 images/       # Static images
├── 📁 templates/        # HTML templates
├── 📁 dataset/          # Student face images
├── 📁 venv/            # Virtual environment
├── 📄 app.py           # Main Flask application
├── 📄 model.py         # ML model training & prediction
├── 📄 requirements.txt # Python dependencies
├── 📄 attendance.db    # SQLite database
└── 📄 README.md        # This file
```

---

## 🔧 Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend** | Flask 3.0.3 | Web framework |
| **AI/ML** | MediaPipe, OpenCV, Scikit-learn | Face detection & recognition |
| **Database** | SQLite | Data persistence |
| **Frontend** | Bootstrap 5, JavaScript | User interface |
| **Language** | Python 3.11+ | Core development |

---

## 📊 Database Schema

### Students Table
- `id` - Primary key
- `name` - Student full name
- `reg_no` - Registration number
- `batch` - Student batch (Batch 1-5)
- `degree` - Degree program
- `created_at` - Registration timestamp

### Attendance Table
- `id` - Primary key
- `student_id` - Foreign key to students
- `name` - Student name (denormalized)
- `module_code` - Course module identifier
- `timestamp` - Attendance timestamp

---

## 🎯 Key Features Explained

### 🔍 Face Recognition Pipeline
1. **Face Detection** - MediaPipe identifies faces in real-time
2. **Feature Extraction** - OpenCV processes face regions
3. **Classification** - Scikit-learn RandomForest predicts student identity
4. **Confidence Threshold** - Only high-confidence matches are accepted

### 🚫 Duplicate Prevention
- Checks for existing attendance records
- Prevents same student from marking attendance twice
- Module-specific and date-specific validation
- User-friendly error messages

### 📈 Module-based Organization
- Separate attendance tracking per module
- Individual CSV exports per module
- Filtered record views
- Organized data management

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Your Name**
- GitHub: [widushan](https://github.com/yourusername)
- LinkedIn: [https://www.linkedin.com/in/pasindu-widushan-865423281/](https://linkedin.com/in/yourprofile)
- Email: widushanp@gmail.com

---

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for face detection
- [OpenCV](https://opencv.org/) for computer vision
- [Flask](https://flask.palletsprojects.com/) for web framework
- [Bootstrap](https://getbootstrap.com/) for UI components

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

[![GitHub stars](https://img.shields.io/github/stars/yourusername/face-recognition-attendance-system.svg?style=social&label=Star)](https://github.com/yourusername/face-recognition-attendance-system)

Made with ❤️ and Python

</div>