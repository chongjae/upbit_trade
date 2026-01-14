FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install tzdata and set timezone
RUN apt-get update && apt-get install -y tzdata
ENV TZ=Asia/Seoul

# Copy application code
COPY . .

# Grant execution permission to start script
RUN chmod +x start.sh

# Expose the configured Flask port
EXPOSE 55555

# Set entrypoint
CMD ["./start.sh"]
