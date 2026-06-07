# This file defines how to build a Docker image for the Streamlit CFO Dashboard.
#
# The goal is to package the app with Python, required libraries, and the run command so it can run the same way on any machine that has Docker installed.

# Use Python 3.11 slim as the base image.
# "slim" is smaller than the full Python image because it removes many extra system tools.
# Smaller images are faster to download, build, and deploy.
FROM python:3.11-slim

# Set the working directory inside the container.
# All following commands will run from /app.
WORKDIR /app

# Install system-level dependencies needed by the container.
# curl is required because the HEALTHCHECK command uses curl to test the Streamlit app.
#
# apt-get update:
#   Refreshes the list of available Linux packages.
#
# apt-get install -y curl:
#   Installs curl without asking for confirmation.
#
# rm -rf /var/lib/apt/lists/*:
#   Cleans temporary package lists to keep the Docker image smaller.
RUN apt-get update \
    && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt first.
# This helps Docker use cache efficiently.
#
# If requirements.txt does not change, Docker can reuse the previous pip install layer.
# This makes future builds faster because dependencies do not need to reinstall every time.
COPY requirements.txt .

# Install Python dependencies from requirements.txt.
#
# --no-cache-dir:
#   Prevents pip from storing downloaded package files inside the image.
#   This keeps the final image size smaller.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container.

COPY . .

# Tell Docker that the container will listen on port 8501.
# Streamlit uses port 8501 by default.
#
# EXPOSE does not publish the port by itself.
# We still need to map the port when running the container using:
# docker run -p 8501:8501 image-name
EXPOSE 8501

# Add a health check for the running container.
# Docker will use this command to check if the Streamlit app is responding.
#
# --interval=30s:
#   Run the health check every 30 seconds.
#
# --timeout=10s:
#   Wait up to 10 seconds for a response.
#
# --retries=3:
#   Mark the container as unhealthy after 3 failed attempts.
#
# Streamlit exposes a health endpoint at:
# http://localhost:8501/_stcore/health
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start the Streamlit application.
#
# streamlit run streamlit/app.py:
#   Runs the main Streamlit app file.
#
# --server.address=0.0.0.0:
#   Makes Streamlit listen on all network interfaces inside the container.
#   This is required so the app can be accessed from outside Docker.
#
# --server.port=8501:
#   Runs Streamlit on port 8501.
#
# --server.headless=true:
#   Prevents Streamlit from trying to open a browser inside the container.
CMD ["streamlit", "run", "streamlit/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]