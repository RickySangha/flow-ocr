import os
import sys
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2

# Add current directory to path so we can import main
sys.path.append(os.getcwd())

# Patch storage.Client BEFORE importing main to avoid auth error
with patch("google.cloud.storage.Client"):
    from main import app, models

from fastapi.testclient import TestClient

client = TestClient(app)

# Mock GCS
@pytest.fixture
def mock_gcs():
    with patch("main.storage.Client") as mock_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        # Mock download_to_filename to create a dummy image
        def side_effect(filename):
            # Create a white image with some black text-like rectangles
            img = np.ones((1000, 800, 3), dtype=np.uint8) * 255
            cv2.rectangle(img, (100, 100), (700, 200), (0, 0, 0), -1) # Header
            cv2.rectangle(img, (100, 300), (700, 400), (0, 0, 0), -1) # Body
            cv2.imwrite(filename, img)
            
        mock_blob.download_to_filename.side_effect = side_effect
        yield mock_client

# Mock Models to avoid loading heavy weights
@pytest.fixture(autouse=True)
def mock_models():
    # Mock Scanify
    mock_scanify = MagicMock()
    mock_scan_result = MagicMock()
    # Return a dummy image
    img = np.ones((1000, 800, 3), dtype=np.uint8) * 255
    mock_scan_result.image_color = img
    mock_scan_result.image_bw = None
    mock_scanify.process.return_value = mock_scan_result
    
    # Mock DocSeg
    mock_docseg = MagicMock()
    mock_seg_result = MagicMock()
    mock_seg_result.abstain = False
    mock_seg_result.regions = [
        MagicMock(label="header", bbox=(100, 100, 700, 200), score=0.9),
        MagicMock(label="body", bbox=(100, 300, 700, 400), score=0.8)
    ]
    mock_docseg.predict.return_value = mock_seg_result
    
    # Mock OCR
    mock_ocr = MagicMock()
    mock_ocr_result1 = MagicMock(segment_type="header", raw_text="Header Text", confidence=0.95, normalized_value="Header Text")
    mock_ocr_result2 = MagicMock(segment_type="body", raw_text="Body Text", confidence=0.85, normalized_value="Body Text")
    mock_ocr.predict.return_value = [mock_ocr_result1, mock_ocr_result2]
    
    # Patch the global models dict in main
    with patch.dict(models, {"scanify": mock_scanify, "docseg": mock_docseg, "ocr": mock_ocr}):
        yield

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_ocr_endpoint(mock_gcs):
    payload = {
        "gcs_path": "gs://my-bucket/my-doc.png",
        "job_id": "test_job_123",
        "doc_type": "form_v1"
    }
    
    response = client.post("/ocr", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "test_job_123"
    assert data["status"] == "success"
    assert "results" in data
    assert len(data["results"]["regions"]) == 2
    assert len(data["results"]["ocr"]) == 2
    assert data["results"]["ocr"][0]["text"] == "Header Text"
