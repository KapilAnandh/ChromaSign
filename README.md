**README.md**
# 🖐️ Sign Language Recognition & Shirt Color Detection
This project combines **Computer Vision (OpenCV + Mediapipe)** and **Transformers (Hugging Face Models)** to perform two key tasks on a given image:
1. **Sign Language Gesture Recognition** — using a pretrained Vision Transformer (ViT) model.
2. **Shirt Color Detection** — using semantic segmentation (SegFormer) and color clustering (KMeans).

## 🚀 Features
- Detects **human hands and landmarks** using Mediapipe.
- Recognizes **sign language gestures** using `google/vit-base-patch16-224`.
- Identifies the **dominant shirt color** by:
  - Performing **semantic segmentation** with NVIDIA’s `segformer-b0-finetuned-ade-512-512`.
  - Extracting the **torso region**.
  - Applying **KMeans clustering** to find the dominant RGB color.
  - Mapping RGB values to the closest **CSS3 color name**.
- Displays results visually and textually.

## 🧩 How It Works

1. **Input:**  
   The user provides an image (e.g., a person performing a sign gesture).

2. **Processing Steps:**
   - **Pose Detection:** Mediapipe estimates human pose landmarks (shoulders, hips, etc.).
   - **Hand Detection:** Mediapipe identifies and draws hand landmarks.
   - **Shirt Segmentation:** SegFormer segments the image to isolate the person’s torso area.
   - **Color Extraction:** KMeans finds the most dominant shirt color.
   - **Gesture Prediction:** The Vision Transformer (ViT) model predicts the sign/gesture class.

3. **Output:**
   - Displays:
     - Hand landmarks overlay on the original image.
     - Detected shirt color patch.
   - Prints:
     - Dominant shirt color name and RGB value.
     - Predicted sign/gesture label.

## 📸 Example Output

👕 Detected Shirt Color: blue (RGB: (42, 68, 173))
🖐️ Predicted Sign/Gesture Label: Thumbs Up

Visuals:
- **Input Image**
- **Detected Hands Overlay**
- **Detected Shirt Color Patch**

## 🛠️ Installation

### 1️⃣ Clone the Repository
git clone https://github.com/<your-username>/sign-language-color-detector.git
cd sign-language-color-detector
### 2️⃣ Install Dependencies
Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate     # (Linux/Mac)
venv\Scripts\activate        # (Windows)
Install all dependencies:

pip install -r requirements.txt
⚙️ Usage
Place your image in the sample_images/ folder, then run:

python app/main.py
To test with a new image:

image_path = "sample_images/your_image.jpg"
**📦 Dependencies**
Python 3.9+
torch
torchvision
transformers
mediapipe
opencv-python-headless
matplotlib
numpy
scikit-learn
pillow
huggingface_hub
(See requirements.txt for exact versions.)

**💡 Where It’s Useful**
Sign Language Recognition Systems — can be extended to detect and classify real-time sign gestures.
Human-Computer Interaction (HCI) — color detection + gesture control for accessibility systems.
Surveillance & Analytics — detecting specific color codes (e.g., uniform colors).
Education & Research — demonstration of multimodal computer vision using segmentation + classification.

🤝 Contributing
Pull requests are welcome!
If you have suggestions for improvements or want to train on custom gestures, please open an issue or submit a PR.

🧑‍💻 Author
Kapil Anandh
