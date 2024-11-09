import cv2
import os
import numpy as np
from random import randint, uniform

def load_images_from_folder(folder_path):
    images = []
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filter common image formats
            full_path = os.path.join(folder_path, filename)
            img = cv2.imread(full_path)
            if img is not None:
                images.append(img)
                image_paths.append(full_path)
    return images, image_paths

def random_flip(img):
    return cv2.flip(img, randint(-1, 1))

def random_rotation(img, angle_range=30):
    angle = uniform(-angle_range, angle_range)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h))
    return rotated

def random_shift(img, shift_max=30):
    dx, dy = randint(-shift_max, shift_max), randint(-shift_max, shift_max)
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
    return shifted

def random_scale(img, scale_range=(0.75, 1.25)):
    scale = uniform(*scale_range)
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    scaled = cv2.resize(img, (new_w, new_h))
    return cv2.resize(scaled, (w, h))

def add_random_noise(img, noise_level=50):
    noise = np.random.randint(-noise_level, noise_level, img.shape, dtype='int16')
    noisy_img = np.clip(img + noise, 0, 255).astype('uint8')
    return noisy_img

def augment_image(img):
    img = random_flip(img)
    img = random_rotation(img)
    # img = random_shift(img)
    # img = random_scale(img)
    img = add_random_noise(img)
    return img

def pad_image(img, target_size=(256, 256), padding_color=(0, 0, 0)):
    """Pads an image to the specified target size while maintaining its aspect ratio."""
    original_h, original_w = img.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scaling factor to fit the image within target size
    scale = min(target_w / original_w, target_h / original_h)
    new_w, new_h = int(original_w * scale), int(original_h * scale)
    
    # Resize the image while preserving aspect ratio
    resized_img = cv2.resize(img, (new_w, new_h))
    
    # Create a new image with the target size and fill with padding color
    padded_img = np.full((target_h, target_w, 3), padding_color, dtype=np.uint8)
    
    # Calculate padding positions
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    padded_img[top:top + new_h, left:left + new_w] = resized_img
    
    return padded_img

def save_augmented_images(images, image_paths, output_folder="augmented", target_size=(1600, 1200)):
    os.makedirs(output_folder, exist_ok=True)
    for idx, (img, path) in enumerate(zip(images, image_paths)):
        padded_img = pad_image(img, target_size)
        cv2.imwrite(os.path.join(output_folder, f"augmented_{idx}_0.jpeg"), padded_img)
        for i in range(1, 10):
            augmented_img = augment_image(padded_img)
            cv2.imwrite(os.path.join(output_folder, f"augmented_{idx}_{i}.jpeg"), augmented_img)
            
        # print(padded_img.shape)
        # filename = os.path.basename(path)
        # new_filename = f"augmented_{idx}_{filename}"
        # cv2.imwrite(os.path.join(output_folder, new_filename), padded_img)
        # cv2.imwrite(os.path.join(output_folder, new_filename), augmented_img)

if __name__ == "__main__":
    folder_path = "/home/tree/Courses/CV/Final_Project/data"  # Replace with the path to your folder containing images
    images, image_paths = load_images_from_folder(folder_path)
    print(f"Found {len(images)} images")
    save_augmented_images(images, image_paths, target_size=(1600, 1200))  # Set desired target size
    folder_path = "/home/tree/Courses/CV/Final_Project/augmented"
    images, image_paths = load_images_from_folder(folder_path)
    print(f"Found {len(images)} augmented images")
