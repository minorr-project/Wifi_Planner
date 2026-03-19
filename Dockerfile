FROM warpdotdev/dev-base:latest

WORKDIR /workspace

# Install Python dependencies as a cacheable layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Ensure output directories exist
RUN mkdir -p outputs/images

# Use the non-interactive Agg backend — no display needed inside the container
ENV MPLBACKEND=Agg

EXPOSE 5000

CMD ["python3", "app.py"]
