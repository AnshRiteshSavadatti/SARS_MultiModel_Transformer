import os
import csv
from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from PIL import Image
import torch

# Set device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the Mini-InternVL model
path = "OpenGVLab/Mini-InternVL-Chat-2B-V1-5"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).eval().to(device)

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
image_processor = CLIPImageProcessor.from_pretrained(path)

# Function to process an image
def process_image(image_path):
    if not os.path.exists(image_path):
        return None
    try:
        image = Image.open(image_path).resize((448, 448))
        pixel_values = image_processor(images=image, return_tensors='pt').pixel_values.to(torch.float16).to(device)
        return pixel_values
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Function to interpret the model's response as "yes" or "no"
def interpret_response(response):
    response = response.strip().lower()
    if 'yes' in response:
        return 'yes'
    elif 'no' in response:
        return 'no'
    else:
        return 'no'

# Calculate similarity based on model answers to questions with updated weights
def calculate_similarity(image1_path, image2_path, questions):
    # Assign weights based on the question index
    weights = [120] + [7] * 10 + [6] * (len(questions) - 11)  # First question weight = 4, next 10 = 2, rest = 1
    total_weight = sum(weights)

    # Process the two images
    pixel_values1 = process_image(image1_path)
    pixel_values2 = process_image(image2_path)

    if pixel_values1 is None or pixel_values2 is None:
        return 0, []

    answers1, answers2 = [], []
    for question in questions:
        try:
            # Get model responses for both images
            response1 = model.chat(tokenizer, pixel_values1, question, dict(max_new_tokens=256, do_sample=False))
            response2 = model.chat(tokenizer, pixel_values2, question, dict(max_new_tokens=256, do_sample=False))

            # Interpret responses and store answers
            answers1.append(interpret_response(response1))
            answers2.append(interpret_response(response2))
        except Exception as e:
            print(f"Error with question '{question}': {e}")
            answers1.append("no")
            answers2.append("no")

    # Calculate weighted similarity score
    weighted_score = sum(
        weight if a1 == a2 else 0
        for a1, a2, weight in zip(answers1, answers2, weights)
    )

    # Normalize the score to a percentage (from 0 to 100)
    similarity_percentage = (weighted_score / total_weight) * 100

    return similarity_percentage, answers1

# Evaluate similarity for all pairs in the CSV file
def evaluate_train_pairs(input_csv, output_csv, base_dir, questions):
    with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.reader(infile)  # Use csv.reader directly (no need for list conversion)
        writer = csv.writer(outfile)

        header = next(reader)  # Read and write the header to output CSV
        writer.writerow(header + ['Similarity'] + [f"Q{i+1}" for i in range(len(questions))])

        for row in reader:
            image1_path = os.path.join(base_dir, row[0])
            image2_path = os.path.join(base_dir, row[1])

            similarity, answers = calculate_similarity(image1_path, image2_path, questions)
            writer.writerow(row + [similarity] + answers)

            print(f"Processed pair: {row[0]} and {row[1]} -> Similarity: {similarity}%")

# Process the train dataset sequentially
def process_train_sequentially(train_csv, train_dir, questions):
    evaluate_train_pairs(train_csv, 'train_results_sequential.csv', train_dir, questions)

# Paths and questions
train_dir = '/workspace/train/images'
train_csv = '/workspace/train2_filtered.csv'  # Updated to train2.csv

questions = [
    "Answer 'yes' or 'no' only: Is the image taken during nighttime (dark environment)?",
    "Answer 'yes' or 'no' only: Are there any vehicles moving in the opposite direction in the image?",
    "Answer 'yes' or 'no' only: Are there any intersections and sharp turns visible in the image that require the driver to yield or stop?",
    "Answer 'yes' or 'no' only: Are there visible potholes, cracks, or damaged road surfaces that could pose a hazard to vehicles?",
    "Answer 'yes' or 'no' only: Is there a high likelihood of vehicles entering the road from visible side streets or driveways?",
    "Answer 'yes' or 'no' only: Does the image depict an urban environment with buildings or city infrastructure?",
    "Answer 'yes' or 'no' only: Does the image show more than 5 vehicles close together on the road, indicating congestion?",
    "Answer 'yes' or 'no' only: Are there visible bridges or flyovers in the image?",
    "Answer 'yes' or 'no' only: Is there visible construction work or a roadblock in the image?",
    "Answer 'yes' or 'no' only: Is there a visible pedestrian crossing or zebra crossing in the image?",
    "Answer 'yes' or 'no' only: Are there visible speed limit signs in the image?",
    "Answer 'yes' or 'no' only: Is there a visible emergency vehicle (e.g., ambulance, fire truck, or police car) in the image?",
    "Answer 'yes' or 'no' only: Are there visible toll booths or entry gates in the image?",
    "Answer 'yes' or 'no' only: Is there visible snowfall or icy conditions on the road?",
    "Answer 'yes' or 'no' only: Are there visible traffic lights in the image?",
    "Answer 'yes' or 'no' only: Are there visible two-wheeled vehicles (motorcycles or bicycles) on the road in the image?",
    "Answer 'yes' or 'no' only: Are there visible road markings (e.g., lane markers, arrows) in the image?",
    "Answer 'yes' or 'no' only: Is the road surface dry in the image?",
    "Answer 'yes' or 'no' only: Are there visible vehicles with their headlights on in the image?",
    "Answer 'yes' or 'no' only: Is there a visible detour or alternative route sign in the image?",
    "Answer 'yes' or 'no' only: Is there limited visibility due to shadows, glare, fog, smoke or oncoming headlights in the image?"
]

# Process the train dataset sequentially for train2.csv
process_train_sequentially(train_csv, train_dir, questions)
