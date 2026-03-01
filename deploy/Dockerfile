# ==============================
# 1️⃣ Base Image (Slim Python)
# ==============================
FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# ==============================
# 2️⃣ Install system dependencies
# ==============================
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ==============================
# 3️⃣ Copy requirements first (Docker cache optimization)
# ==============================
COPY requirements.txt .

# Upgrade pip and install deps
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ==============================
# 4️⃣ Copy project files
# ==============================
COPY . .

# ==============================
# 4️⃣ Create non-root user
# ==============================
RUN useradd -m appuser

# ==============================
# 5️⃣ Streamlit config (Production mode)
# ==============================
RUN mkdir -p /home/appuser/.streamlit && \
    echo "\
[server]\n\
headless = true\n\
port = 8080\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
" > /home/appuser/.streamlit/config.toml

USER appuser

# ==============================
# 6️⃣ Expose port
# ==============================
EXPOSE 8080

# ==============================
# 7️⃣ Healthcheck (optional but recommended)
# ==============================
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health || exit 1

# ==============================
# 8️⃣ Run Streamlit
# ==============================
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
