import cv2
import mediapipe as mp
import numpy as np
import torch
from sklearn.cluster import KMeans
from transformers import AutoProcessor, AutoModelForImageClassification, AutoImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import matplotlib.pyplot as plt

# ===== CSS3 Color Map =====
CSS3_COLORS = {
    "black": (0,0,0), "white": (255,255,255), "red": (255,0,0), "lime": (0,255,0),
    "blue": (0,0,255), "yellow": (255,255,0), "cyan": (0,255,255), "magenta": (255,0,255),
    "silver": (192,192,192), "gray": (128,128,128), "maroon": (128,0,0), "olive": (128,128,0),
    "green": (0,128,0), "purple": (128,0,128), "teal": (0,128,128), "navy": (0,0,128)
}

def closest_color_name(rgb):
    min_dist, closest = float('inf'), None
    for name, color in CSS3_COLORS.items():
        dist = sum((rgb[i]-color[i])**2 for i in range(3))
        if dist < min_dist:
            min_dist, closest = dist, name
    return closest

def detect_shirt_color(image_path):
    image = Image.open(image_path).convert("RGB")
    np_img = np.array(image)

    # Load segmentation model
    processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.cpu()
    upsampled_logits = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()

    person_mask = (pred_seg == 12).astype(np.uint8)
    h, w = person_mask.shape
    torso_mask = np.zeros_like(person_mask)
    torso_mask[int(h*0.4):int(h*0.85), :] = person_mask[int(h*0.4):int(h*0.85), :]

    shirt_pixels = np_img[torso_mask == 1]
    if len(shirt_pixels) == 0:
        return "unknown", (128,128,128)

    kmeans = KMeans(n_clusters=1, n_init=10)
    kmeans.fit(shirt_pixels)
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    color_name = closest_color_name(dominant_color)

    return color_name, tuple(dominant_color)

def recognize_sign(image_path):
    processor = AutoProcessor.from_pretrained("google/vit-base-patch16-224")
    model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

    inputs = processor(images=Image.open(image_path), return_tensors="pt")
    outputs = model(**inputs)
    preds = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label = model.config.id2label[int(torch.argmax(preds))]
    return label

if __name__ == "__main__":
    image_path = "sample_images/signlanguage_sample.jpg"

    # Load and display original image
    image_cv = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_image)
    plt.title("Input Image")
    plt.axis("off")
    plt.show()

    # Hand Detection
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        results = hands.process(rgb_image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(rgb_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    plt.imshow(rgb_image)
    plt.title("Detected Hands")
    plt.axis("off")
    plt.show()

    # Run shirt color detection
    shirt_color_name, dominant_color = detect_shirt_color(image_path)

    # Run sign recognition
    sign_label = recognize_sign(image_path)

    # Display results
    plt.imshow([[np.array(dominant_color)/255]])
    plt.title(f"Shirt Color: {shirt_color_name}")
    plt.axis("off")
    plt.show()

    print(f"Detected Shirt Color: {shirt_color_name} (RGB: {dominant_color})")
    print(f"Predicted Sign/Gesture Label: {sign_label}")
