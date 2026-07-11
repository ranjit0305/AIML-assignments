import os
import cv2
import pandas as pd

from sklearn.model_selection import train_test_split

dataset_path = "PetImages"
processed_path = "processed_images"

# Create output folder
os.makedirs(processed_path, exist_ok=True)

# Lists to store metadata
image_paths = []
labels = []

print(os.listdir(dataset_path))

for label in os.listdir(dataset_path):

    label_folder = os.path.join(dataset_path, label)

    if not os.path.isdir(label_folder):
        continue

    print(f"Processing {label}...")

    for image_name in os.listdir(label_folder):

        image_path = os.path.join(label_folder, image_name)

        try:
            image = cv2.imread(image_path)

            if image is None:
                continue

            # Resize
            image = cv2.resize(image, (128, 128))

            # Normalize
            image = image / 255.0

            # Save processed image
            output_path = os.path.join(processed_path, image_name)

            cv2.imwrite(
                output_path,
                (image * 255).astype("uint8")
            )

            image_paths.append(output_path)
            labels.append(label)

        except Exception as e:
            print(f"Skipping {image_name}: {e}")

print("\nProcessing Complete!")
print("Total Images:", len(image_paths))

# Create Metadata CSV
metadata = pd.DataFrame({
    "ImagePath": image_paths,
    "Label": labels
})

metadata.to_csv("metadata.csv", index=False)

print("\nMetadata Created Successfully!")
print(metadata.head())

# Split into Train and Validation

train_df, validation_df = train_test_split(
    metadata,
    test_size=0.2,
    random_state=42
)

print("\nDataset Split Successfully!")

print("Training Images:", len(train_df))
print("Validation Images:", len(validation_df))