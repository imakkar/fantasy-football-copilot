# Fantasy Football Co-Pilot -- reproducible environment
#
# nfl_data_py pins pandas<2.0, which requires Python 3.9-3.11, so we pin 3.11.
#
# Build:  docker build -t ff-copilot .
# Build the vector store (no API key needed):
#   docker run --rm -v "$PWD/data:/app/data" ff-copilot python src/build_index.py
# Run the pipeline (Gemini free tier):
#   docker run --rm -e GEMINI_API_KEY=$GEMINI_API_KEY \
#     -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" \
#     ff-copilot python src/model_runner.py

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project.
COPY . .

# Default command shows the help; override with build_index / model_runner.
CMD ["python", "src/model_runner.py", "--help"]
