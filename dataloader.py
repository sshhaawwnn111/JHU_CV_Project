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

def add_random_noise(img, noise_level=30):
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

def save_augmented_images(images, image_paths, output_folder="Broken"):
    p = os.path.join("augmented", output_folder)
    os.makedirs(p, exist_ok=True)
    for idx, (img, path) in enumerate(zip(images, image_paths)):
        for i in range(10):
            augmented_img = augment_image(img)
            # filename = os.path.basename(path)
            new_filename = f"{idx}_{i}.jpeg"
            cv2.imwrite(os.path.join(p, new_filename), augmented_img)

if __name__ == "__main__":
    foler_name = "Missing_Tablets"
    folder_path = f"/home/tree/Courses/CV/Final_Project/{foler_name}"  # Replace with the path to your folder containing images
    images, image_paths = load_images_from_folder(folder_path)
    print(f"Loaded {len(images)} images from {folder_path}")
    save_augmented_images(images, image_paths, output_folder=foler_name)




    folder_path = f"/home/tree/Courses/CV/Final_Project/augmented/{foler_name}"  # Replace with the path to your folder containing images
    images, image_paths = load_images_from_folder(folder_path)
    print(f"Saved {len(images)} images")




