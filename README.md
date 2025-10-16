# NeuralNetCar4

## Overview
This project is a locally-running, web-specific, **DNN+RL (Deep Neural Network + Reinforcement Learning)** autonomous obstacle avoidance agent.

## Setup and Requirements
All code files are prepared and should be placed within the same folder hierarchy. Please place them inside a folder named "NeuralNetCar4".
A virtual environment is required, and libraries such as TensorFlow need to be installed.

## 📂 Folder Structure

```
NeuralNetCar4/
├── index.html
├── simulation.js
├── main_server_refactored.py (Python Backend)
├── venv (Virtual Environment folder)
└── model/ (Stores AI weights)
```

## Execution Guide

### Command-Line Setup
First, navigate into the folder hierarchy:
```bash
C:
cd \---------\NeuralNetCar4
```

### 1. Create a Virtual Environment
```bash
python -m venv venv
```

### 2. Activate the Virtual Environment
```bash
venv\Scripts\activate.bat
```

### 3. Install Required Libraries
```bash
pip install tensorflow numpy
```

### 4. Execute the Server
```bash
python main_server_refactored.py
```

### 5. Access the Web Interface
Open the HTML file in a browser:
```
file:///C:/---------/NeuralNetCar4/index.html
```

## How to Stop the Process ?
To stop the process:
Press **Ctrl+C**
*Do not press it multiple times. Wait patiently without panicking.*
