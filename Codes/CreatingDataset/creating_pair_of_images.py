import os
import csv

def create_image_pairs_sequential(train_dir, test_dir, val_dir, output_dir):
    """
    Creates sequential pairs of images from the specified directories and stores them in separate CSV files in the output directory.

    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    def generate_sequential_pairs(image_dir, output_csv):
        # List all images in the directory
        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))])

        # Ensure there are enough images
        if not image_files:
            print(f"No images found in {image_dir}. Skipping...")
            return

        # Handle odd number of images by pairing the last image with itself
        if len(image_files) % 2 != 0:
            image_files.append(image_files[-1])

        # Create sequential pairs
        pairs = [(image_files[i], image_files[i + 1]) for i in range(0, len(image_files), 2)]

        # Save pairs to a CSV file in the output directory
        output_path = os.path.join(output_dir, output_csv)
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Image 1', 'Image 2'])
            writer.writerows(pairs)

        print(f"Pairs saved to {output_path}")

    # Generate pairs for train, test, and validation directories
    generate_sequential_pairs(train_dir, 'train_image_pairs.csv')
    generate_sequential_pairs(test_dir, 'test_image_pairs.csv')
    generate_sequential_pairs(val_dir, 'validation_image_pairs.csv')


# Paths to the directories
train_dir = '/workspace/train/images'
test_dir = '/workspace/test'
val_dir = '/workspace/val/images'
output_dir = '/workspace/csvDataset'

# Create image pairs
create_image_pairs_sequential(train_dir, test_dir, val_dir, output_dir)