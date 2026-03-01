# 🎙️ AI Voice Detection -- Deployment Guide

This folder contains the production-ready deployment setup for the AI
Speech Detection (Human vs AI-Generated Speech) system.

It includes: - Streamlit app - Trained models - Docker configuration -
Local launch scripts - Cloud Run deployment steps

------------------------------------------------------------------------

# 🖥️ Run Locally (Without Docker)

## ✅ Windows

Double-click:

launch.bat

OR from terminal:

launch.bat

------------------------------------------------------------------------

## ✅ macOS / Linux

Make executable once:

chmod +x run.sh

Then run:

./run.sh

The app will start at:

http://localhost:8501

------------------------------------------------------------------------

# 🐳 Run With Docker (Local Production Simulation)

## 1️⃣ Build image

docker build -t voiceauth .

## 2️⃣ Run container

docker run -p 8080:8080 voiceauth

Open:

http://localhost:8080

------------------------------------------------------------------------

# ☁️ Deploy to Google Cloud Run

## 🔹 Step 1: Authenticate

gcloud auth login

------------------------------------------------------------------------

## 🔹 Step 2: Set Project

gcloud config set project YOUR_PROJECT_ID

Verify:

gcloud config get-value project

------------------------------------------------------------------------

## 🔹 Step 3: Enable Required APIs (first time only)

gcloud services enable run.googleapis.com artifactregistry.googleapis.com

------------------------------------------------------------------------

## 🔹 Step 4: Create Artifact Registry (one-time setup)

gcloud artifacts repositories create voice-auth --repository-format=docker --location=northamerica-northeast1

------------------------------------------------------------------------

## 🔹 Step 5: Build Docker Image Locally

docker build -t voiceauth .

------------------------------------------------------------------------

## 🔹 Step 6: Tag Image

docker tag voiceauth northamerica-northeast1-docker.pkg.dev/voiceauth-488102/voice-auth/voice-auth

------------------------------------------------------------------------

## 🔹 Step 7: Push Image

gcloud auth configure-docker northamerica-northeast1-docker.pkg.dev
docker push northamerica-northeast1-docker.pkg.dev/voiceauth-488102/voice-auth/voice-auth

------------------------------------------------------------------------

## 🔹 Step 8: Deploy to Cloud Run

gcloud builds submit . --tag northamerica-northeast1-docker.pkg.dev/voiceauth-488102/voice-auth/voice-auth   

gcloud run deploy voice-auth --image northamerica-northeast1-docker.pkg.dev/voiceauth-488102/voice-auth/voice-auth --region northamerica-northeast1 --platform managed --allow-unauthenticated --memory 2Gi --cpu 1

After deployment, Cloud Run will provide a public HTTPS URL.

------------------------------------------------------------------------

# ⚙️ Recommended Cloud Run Settings

For CNN-LSTM model:

-   Memory: 4Gi minimum
-   CPU: 2
-   Concurrency: 1 (for consistent latency)
-   Timeout: 300 seconds

------------------------------------------------------------------------

# 🏗️ Deployment Architecture

User → Cloud Run (Dockerized Streamlit App) → Model Inference

-   Stateless container
-   Auto-scaling enabled
-   Secure HTTPS endpoint

------------------------------------------------------------------------

# 🛠️ Troubleshooting

### If build is very large:

Ensure datasets are NOT inside this deploy folder.

### If memory error:

Increase Cloud Run memory to 8Gi.

### If model not found:

Ensure outputs/models/ contains: - cnn_best.pt - cnn_lstm_best.pt -
svm.joblib - random_forest.joblib - gradient_boosting.joblib

------------------------------------------------------------------------

# 🎓 Academic Note

This deployment setup demonstrates:

-   Containerized ML deployment
-   Cloud-native inference
-   Auto-scaling serverless architecture
-   Production-grade dependency management

Suitable for research demos and academic evaluation.

------------------------------------------------------------------------

# 👤 Author

Rajib Roy\
M.Tech AI Project\
AI-Generated Speech Detection System
