# Real-Time-Steel-Defect-and-Coating-Detection
A real-time industrial inspection system for zinc-coated steel using OpenCV, LBP texture analysis, and IDS industrial cameras. Automatically detects surface defects and coating inconsistencies, and controls hardware relays via serial communication for automated quality control.


This project leverages advanced image processing and machine vision techniques to:
- Detect the presence and outline of steel sheets on a production line
- Evaluate coating consistency using **Local Binary Pattern (LBP)** texture analysis
- Enhance surface imaging using **CLAHE** and identify physical defects through contour analysis
- Trigger **relay-based hardware control** via serial communication based on inspection results
- Operate in real-time using multithreaded camera acquisition and processing
