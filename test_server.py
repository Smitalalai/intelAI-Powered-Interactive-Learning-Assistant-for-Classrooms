#!/usr/bin/env python3
"""
Simple test server to verify FastAPI setup
"""
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Test Learning Assistant API")

@app.get("/")
async def root():
    return {"message": "AI Learning Assistant Test API is running!"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "Test server is working"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
