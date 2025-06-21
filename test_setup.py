import requests
import json
import os
import glob
from pathlib import Path

def get_latest_image():
    """Get the latest image file from the uploads folder"""
    uploads_dir = "uploads"
    
    if not os.path.exists(uploads_dir):
        print(f"❌ Uploads directory '{uploads_dir}' does not exist")
        return None
    
    # Get all image files from uploads folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(uploads_dir, ext)))
        image_files.extend(glob.glob(os.path.join(uploads_dir, ext.upper())))
    
    if not image_files:
        print(f"❌ No image files found in '{uploads_dir}' folder")
        return None
    
    # Get the latest file by modification time
    latest_file = max(image_files, key=os.path.getmtime)
    filename = os.path.basename(latest_file)
    
    print(f"📁 Found {len(image_files)} image(s) in uploads folder")
    print(f"📸 Using latest image: {filename}")
    
    return filename

def test_flask_backend():
    """Test the Flask backend API endpoints"""
    
    print("Testing Flask backend...")
    
    # Get the latest image from uploads folder
    filename = get_latest_image()
    
    if not filename:
        print("❌ No image available for testing")
        return
    
    # Test the analyze endpoint
    test_data = {
        "filename": filename
    }
    
    try:
        response = requests.post(
            "http://localhost:5000/analyze",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Flask backend is working!")
            print(f"📊 Face Shape: {result.get('face_shape', 'Unknown')}")
            print(f"📏 Face Length: {result.get('face_length', 0):.3f}")
            print(f"📐 Cheekbone Width: {result.get('cheekbone_width', 0):.3f}")
            print(f"🖼️ Processed Image: {result.get('processed_image', 'None')}")
            print(f"Response structure: {list(result.keys())}")
        else:
            print(f"❌ Error response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Flask backend. Make sure it's running on localhost:5000")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_flask_backend() 