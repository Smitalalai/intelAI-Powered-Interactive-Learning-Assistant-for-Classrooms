# Deployment Guide - AI-Powered Interactive Learning Assistant

This guide provides comprehensive instructions for deploying the AI-Powered Interactive Learning Assistant for the OpenVINO Unnati Hackathon 2025.

## 🚀 Quick Start Deployment

### Prerequisites

- Python 3.8 or higher
- Intel CPU (recommended for OpenVINO optimization)
- 8GB+ RAM (16GB recommended)
- 10GB+ free disk space
- Internet connection for model downloads

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd intelAI-Powered-Interactive-Learning-Assistant-for-Classrooms

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Setup

```bash
# Run the model setup script
python scripts/setup_models.py

# This will:
# - Download AI models (BLIP, BART, DialoGPT, Whisper)
# - Convert models to OpenVINO format
# - Create sample data
# - Run initial benchmarks
```

### 3. Configuration

```bash
# Copy and customize configuration
cp configs/config.py configs/config_local.py

# Edit configuration as needed
# - Adjust model paths
# - Set device preferences (CPU/GPU)
# - Configure API settings
```

### 4. Run the Demo

```bash
# Run comprehensive demo
python demo.py

# This demonstrates all features:
# - Question answering
# - Summarization
# - Image captioning
# - Multimodal interaction
# - Performance monitoring
```

### 5. Start the Services

```bash
# Terminal 1: Start FastAPI backend
python -m src.api.main

# Terminal 2: Start Streamlit frontend
streamlit run src/ui/app.py

# Access the application:
# - Frontend: http://localhost:8501
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

## 🏗️ Production Deployment

### Docker Deployment

```bash
# Build Docker image
docker build -t learning-assistant .

# Run with docker-compose
docker-compose up -d

# Services will be available at:
# - Frontend: http://localhost:8501
# - API: http://localhost:8000
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n learning-assistant

# Access via LoadBalancer or Ingress
kubectl get svc -n learning-assistant
```

### Cloud Deployment (AWS/Azure/GCP)

#### AWS EC2 Deployment

```bash
# Launch EC2 instance (recommended: c5.2xlarge or higher)
# - AMI: Amazon Linux 2 or Ubuntu 20.04
# - Security groups: Allow ports 8000, 8501

# SSH to instance and run setup
ssh -i your-key.pem ec2-user@your-instance-ip
sudo yum update -y
sudo yum install -y python3 python3-pip git

# Clone and setup as in Quick Start
git clone <repository-url>
cd intelAI-Powered-Interactive-Learning-Assistant-for-Classrooms
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/setup_models.py

# Install PM2 for process management
npm install -g pm2

# Start services with PM2
pm2 start ecosystem.config.js
pm2 save
pm2 startup
```

#### Azure Container Instances

```bash
# Create resource group
az group create --name learning-assistant-rg --location eastus

# Deploy container
az container create \
  --resource-group learning-assistant-rg \
  --name learning-assistant \
  --image your-registry/learning-assistant:latest \
  --cpu 4 \
  --memory 8 \
  --ports 8000 8501 \
  --dns-name-label learning-assistant-demo
```

## 🔧 Configuration Options

### Model Configuration

```python
# configs/config.py
MODEL_CONFIG = {
    "question_answering": {
        "model_name": "microsoft/DialoGPT-medium",
        "max_length": 512,
        "temperature": 0.7,
        "device": "CPU"  # or "GPU" if available
    },
    "image_captioning": {
        "model_name": "Salesforce/blip-image-captioning-base",
        "max_length": 50,
        "device": "CPU"
    }
    # ... other models
}
```

### OpenVINO Optimization

```python
# Enable OpenVINO optimization
OPENVINO_CONFIG = {
    "enabled": True,
    "device": "CPU",  # CPU, GPU, MYRIAD, HDDL
    "precision": "FP32",  # FP32, FP16, INT8
    "optimization_level": "PERFORMANCE",
    "cache_dir": "models/openvino_cache"
}
```

### API Configuration

```python
# API server settings
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": False,
    "workers": 4,
    "log_level": "info"
}
```

## 📊 Performance Optimization

### Hardware Recommendations

**Minimum Requirements:**
- CPU: Intel Core i5 or AMD Ryzen 5
- RAM: 8GB
- Storage: 10GB SSD
- Network: Broadband internet

**Recommended:**
- CPU: Intel Core i7/i9 or AMD Ryzen 7/9
- RAM: 16GB+
- Storage: 50GB+ NVMe SSD
- GPU: Intel integrated graphics or dedicated GPU

**Optimal (Production):**
- CPU: Intel Xeon or AMD EPYC
- RAM: 32GB+
- Storage: 100GB+ NVMe SSD
- GPU: Intel Arc or dedicated GPU
- Network: Gigabit ethernet

### OpenVINO Optimization Tips

1. **Model Optimization:**
   ```bash
   # Convert models to OpenVINO IR format
   mo --input_model model.onnx --output_dir openvino_models/
   
   # Use INT8 quantization for better performance
   pot -c quantization_config.json
   ```

2. **Runtime Optimization:**
   ```python
   # Use optimal number of inference threads
   ie.set_config({"CPU_THREADS_NUM": "8"}, "CPU")
   
   # Enable CPU pinning
   ie.set_config({"CPU_BIND_THREAD": "YES"}, "CPU")
   ```

3. **Memory Optimization:**
   ```python
   # Enable memory reuse
   ie.set_config({"CPU_THROUGHPUT_STREAMS": "1"}, "CPU")
   ```

## 🔒 Security Configuration

### Basic Security

```bash
# Set environment variables
export SECRET_KEY="your-secret-key-here"
export DATABASE_URL="postgresql://user:pass@localhost/db"

# Use HTTPS in production
export USE_HTTPS=true
export SSL_CERT_PATH="/path/to/cert.pem"
export SSL_KEY_PATH="/path/to/key.pem"
```

### API Authentication

```python
# Enable API authentication
SECURITY_CONFIG = {
    "require_auth": True,
    "jwt_secret": "your-jwt-secret",
    "session_timeout": 3600
}
```

### Network Security

```bash
# Configure firewall (Ubuntu/Debian)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 8000/tcp  # API
sudo ufw allow 8501/tcp  # Frontend
sudo ufw enable

# For production, use reverse proxy (nginx)
sudo apt install nginx
sudo systemctl enable nginx
sudo systemctl start nginx
```

## 📈 Monitoring and Logging

### Application Monitoring

```python
# Enable detailed logging
LOGGING_CONFIG = {
    "level": "INFO",
    "file": "logs/app.log",
    "max_size": "100MB",
    "backup_count": 5
}
```

### Performance Monitoring

```bash
# Install monitoring tools
pip install prometheus-client grafana-api

# Start monitoring
python monitoring/start_monitoring.py
```

### Health Checks

```bash
# API health check
curl http://localhost:8000/health

# Expected response:
{
    "status": "healthy",
    "timestamp": 1704067200.0,
    "models": {
        "qa": {"status": "loaded", "device": "cpu"},
        "summarization": {"status": "loaded", "device": "cpu"}
    }
}
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_api.py -v          # API tests
pytest tests/test_learning_assistant.py -v  # Core tests

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Load Testing

```bash
# Install load testing tools
pip install locust

# Run load tests
locust -f tests/load_test.py --host=http://localhost:8000
```

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Fails:**
   ```bash
   # Check available disk space
   df -h
   
   # Verify internet connection
   ping huggingface.co
   
   # Clear model cache
   rm -rf ~/.cache/huggingface/
   rm -rf models/
   python scripts/setup_models.py
   ```

2. **OpenVINO Not Working:**
   ```bash
   # Verify OpenVINO installation
   python -c "import openvino; print(openvino.__version__)"
   
   # Check CPU support
   python -c "from openvino.runtime import Core; print(Core().available_devices)"
   ```

3. **Memory Issues:**
   ```bash
   # Monitor memory usage
   htop
   
   # Reduce batch size in config
   MODEL_CONFIG["batch_size"] = 1
   
   # Enable swap if needed
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

4. **API Connection Issues:**
   ```bash
   # Check if ports are open
   netstat -tlnp | grep 8000
   
   # Test API directly
   curl -X GET http://localhost:8000/
   
   # Check firewall
   sudo ufw status
   ```

### Debug Mode

```bash
# Run in debug mode
export DEBUG=true
export LOG_LEVEL=DEBUG
python -m src.api.main
```

### Log Analysis

```bash
# View recent logs
tail -f logs/app.log

# Search for errors
grep ERROR logs/app.log

# Monitor performance
grep "processing_time" logs/app.log | tail -20
```

## 📋 Maintenance

### Regular Updates

```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Update models (monthly)
python scripts/update_models.py

# Backup configuration
cp -r configs/ backups/configs_$(date +%Y%m%d)
```

### Performance Monitoring

```bash
# Weekly performance check
python scripts/benchmark_models.py

# Monitor disk usage
du -sh models/ logs/ data/

# Clean up old logs
find logs/ -name "*.log" -mtime +30 -delete
```

### Database Maintenance

```bash
# Backup database
pg_dump learning_assistant > backup_$(date +%Y%m%d).sql

# Optimize database
VACUUM ANALYZE;
```

## 🎯 Hackathon Demo Setup

### Quick Demo Environment

```bash
# One-command setup for demo
./scripts/demo_setup.sh

# This script:
# 1. Sets up environment
# 2. Downloads lightweight models
# 3. Creates sample data
# 4. Starts services
# 5. Opens browser to demo
```

### Demo Data

```bash
# Generate demo scenarios
python scripts/generate_demo_data.py

# Creates:
# - Sample questions and answers
# - Educational images
# - Audio samples
# - Lesson content
```

### Presentation Mode

```bash
# Start in presentation mode
export DEMO_MODE=true
python demo.py

# Features:
# - Auto-advance demos
# - Simplified output
# - Performance highlights
# - Visual indicators
```

## 📞 Support

For deployment support:
- Check the troubleshooting section above
- Review logs in `logs/` directory
- Run diagnostic script: `python scripts/diagnose.py`
- Submit issues on GitHub repository

## 🏆 Hackathon Submission

For OpenVINO Unnati Hackathon 2025:
1. Ensure all features are working
2. Run the complete demo: `python demo.py`
3. Capture performance metrics
4. Document OpenVINO optimizations
5. Prepare presentation materials
