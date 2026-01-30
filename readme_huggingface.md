---
title: Fire Detection System
emoji: 🔥
colorFrom: red
colorTo: orange
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

# 🔥 Real-Time Fire Detection System

An AI-powered fire and smoke detection system using MobileNetV2 for real-time video analysis.

## Features

- 📹 **Real-time webcam detection**
- 📁 **Video file upload support**
- 🖼️ **Static image analysis**
- 🔊 **Audio alarm system**
- ⚡ **Optimized performance** with frame skipping
- 🐛 **Debug mode** for model insights

## How to Use

1. **Select Input Source**: Choose between Webcam, Upload Video, or Upload Image
2. **Adjust Settings**: 
   - Set confidence threshold (lower = more sensitive)
   - Configure frame skip rate for performance
   - Enable/disable sound alarms
3. **Start Detection**: Click the appropriate button to begin analysis

## Model Information

- **Architecture**: MobileNetV2
- **Task**: Binary Classification (Fire/No Fire)
- **Input Size**: 128x128 RGB images
- **Validation Accuracy**: 94%

## Performance Tips

- Increase "Process Every N Frames" for faster processing
- Reduce "Display Size" if experiencing lag
- Use frame skipping (3-5) for smooth real-time performance

## Requirements

Make sure your model file `best_mobile_accuracy_model.pth` is in the same directory as `app.py`.
