import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import Image as ReportImage
from reportlab.lib.utils import ImageReader
from datetime import datetime
from tqdm import tqdm
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tumor_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# === Configuration and CLI Arguments ===
def parse_arguments():
    parser = argparse.ArgumentParser(description='Brain Tumor AI Report Generator')
    parser.add_argument('--model', type=str, default="Brain_Tumor_Detection_Model.keras", 
                        help='Path to trained model file')
    parser.add_argument('--test-folder', type=str, default="BrainTumorDataset/Testing", 
                        help='Path to test dataset folder')
    parser.add_argument('--output', type=str, default="AI_Reports", 
                        help='Output folder for reports')
    parser.add_argument('--img-size', type=int, default=128, 
                        help='Image size for model input')
    parser.add_argument('--batch', action='store_true', 
                        help='Process in batch mode without confirmation prompts')
    return parser.parse_args()

# === Image Preprocessing ===
def preprocess(img, img_size):
    """Preprocess image for model prediction"""
    # Keep a copy of original for display
    original = img.copy()
    
    # Resize to model input size
    img = cv2.resize(img, (img_size, img_size))
    
    # Normalize
    img = img / 255.0
    
    return original, np.expand_dims(img, axis=0)

# === Advanced Tumor Segmentation ===
def segment_tumor(img):
    """Segment tumor from brain MRI using adaptive thresholding"""
    # Convert to grayscale if not already
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Use adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Find potential tumor regions (white areas)
    contours, _ = cv2.findContours(sure_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area to find significant regions
    mask = np.zeros_like(gray)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 10000:  # Adjust these thresholds based on your images
            cv2.drawContours(mask, [cnt], 0, 255, -1)
    
    return mask

# === Tumor Growth Simulation ===
def simulate_growth(mask, growth_factor=1.5):
    """Simulate tumor growth using morphological operations"""
    # Convert to binary if not already
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Calculate initial area
    initial_area = np.count_nonzero(binary)
    
    # Simulate growth using dilation
    kernel_size = int(3 * growth_factor)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    grown_mask = cv2.dilate(binary, kernel, iterations=2)
    
    # Calculate grown area
    grown_area = np.count_nonzero(grown_mask)
    
    # Calculate growth percentage
    if initial_area > 0:
        growth_percent = ((grown_area - initial_area) / initial_area) * 100
    else:
        growth_percent = 0
    
    return grown_mask, growth_percent

# === Create Visualization for Tumor Cases ===
def create_visualization(original, day1_mask, day30_mask):
    """Create side-by-side visualization with overlays for tumor cases"""
    # Create a figure for visualization
    plt.figure(figsize=(15, 5))
    
    # Original with Day 1 mask overlay
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original MRI")
    plt.axis('off')
    
    # Day 1 mask
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    mask_overlay = np.zeros_like(original)
    mask_overlay[:,:,1] = day1_mask  # Green channel for Day 1
    plt.imshow(mask_overlay, alpha=0.4)
    plt.title("Day 1 - Detected Tumor")
    plt.axis('off')
    
    # Day 30 mask (simulated growth)
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    grown_overlay = np.zeros_like(original)
    grown_overlay[:,:,0] = day30_mask  # Red channel for Day 30
    plt.imshow(grown_overlay, alpha=0.4)
    plt.title("Day 30 - Projected Growth")
    plt.axis('off')
    
    # Save figure to temporary files in both formats
    viz_path_png = "temp_visualization.png"
    viz_path_jpg = "temp_visualization.jpg"
    plt.savefig(viz_path_png, dpi=150, bbox_inches='tight', format='png')
    plt.savefig(viz_path_jpg, dpi=150, bbox_inches='tight', format='jpeg')
    plt.close()
    
    return viz_path_png, viz_path_jpg

# === Create Visualization for No Tumor Cases ===
def create_no_tumor_visualization(original):
    """Create visualization for cases where no tumor is detected"""
    # Create a figure for visualization
    plt.figure(figsize=(15, 5))
    
    # Original MRI
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original MRI")
    plt.axis('off')
    
    # No tumor detected
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("No Tumor Detected")
    plt.axis('off')
    
    # Healthy brain - future outlook
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Healthy Brain - No Projected Growth")
    plt.axis('off')
    
    # Save figure to temporary files in both formats
    viz_path_png = "temp_visualization.png"
    viz_path_jpg = "temp_visualization.jpg"
    plt.savefig(viz_path_png, dpi=150, bbox_inches='tight', format='png')
    plt.savefig(viz_path_jpg, dpi=150, bbox_inches='tight', format='jpeg')
    plt.close()
    
    return viz_path_png, viz_path_jpg

# === PDF Report Generator ===
def generate_report(image_path, label, confidence, growth_percent, viz_img_paths, output_path, risk_level=None):
    """Generate a comprehensive PDF report"""
    # Set up the canvas
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    
    # Add header with logo
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height-50, "Brain Tumor Analysis Report")
    
    # Add horizontal line
    c.setStrokeColor(colors.grey)
    c.line(50, height-70, width-50, height-70)
    
    # Patient information
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height-100, "Patient Information")
    c.setFont("Helvetica", 12)
    patient_id = os.path.splitext(os.path.basename(image_path))[0]
    c.drawString(50, height-120, f"Patient ID: {patient_id}")
    c.drawString(50, height-140, f"Scan Date: {datetime.now().strftime('%Y-%m-%d')}")
    c.drawString(50, height-160, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Diagnosis information
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height-200, "AI Diagnosis")
    c.setFont("Helvetica", 12)
    
    # Color-code the diagnosis based on tumor type
    if "no_tumor" in label.lower():
        c.setFillColor(colors.green)
    else:
        c.setFillColor(colors.red)
    
    c.drawString(50, height-220, f"Diagnosis: {label.replace('_', ' ').title()}")
    c.setFillColor(colors.black)  # Reset color
    
    c.drawString(50, height-240, f"Model Confidence: {confidence:.2f}%")
    
    # Growth projection - only show for actual tumors
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height-280, "Tumor Growth Projection")
    c.setFont("Helvetica", 12)
    
    if "no_tumor" in label.lower():
        c.drawString(50, height-300, "Estimated 30-Day Growth: N/A (No tumor detected)")
        c.drawString(50, height-320, "Growth Risk Level: None")
    else:
        c.drawString(50, height-300, f"Estimated 30-Day Growth: {growth_percent:.1f}%")
        c.drawString(50, height-320, f"Growth Risk Level: {risk_level}")
    
    # Add visualization image with error handling
    viz_img_path_png, viz_img_path_jpg = viz_img_paths
    
    # Try PNG first, then JPG if PNG fails
    img_added = False
    img_width = width - 100  # Keep margins
    img_height = img_width / 3  # Maintain aspect ratio
    
    try:
        viz_img = ImageReader(viz_img_path_png)
        c.drawImage(viz_img, 50, height-540, width=img_width, height=img_height)
        img_added = True
    except Exception as e:
        logger.warning(f"Failed to read PNG visualization: {e}")
        try:
            viz_img = ImageReader(viz_img_path_jpg)
            c.drawImage(viz_img, 50, height-540, width=img_width, height=img_height)
            img_added = True
        except Exception as e:
            logger.warning(f"Failed to read JPG visualization: {e}")
    
    # If both image formats fail, add a text placeholder
    if not img_added:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, height-400, "Visualization could not be rendered.")
        c.drawString(50, height-420, f"Original diagnosis: {label.replace('_', ' ').title()}")
        if "no_tumor" not in label.lower():
            c.drawString(50, height-440, f"Growth projection: {growth_percent:.1f}%")
    
    # Add disclaimer
    c.setFont("Helvetica", 9)
    disclaimer = """DISCLAIMER: This is an AI-generated report for research purposes only. 
    The growth simulation is based on computer algorithms and should not be used for clinical decision-making.
    Always consult with a qualified medical professional for proper diagnosis and treatment."""
    
    text_object = c.beginText(50, 50)
    text_object.setFont("Helvetica", 9)
    
    # Handle line wrapping manually for the disclaimer
    words = disclaimer.split()
    line = ""
    for word in words:
        test_line = line + word + " "
        # Check if line is getting too long
        if c.stringWidth(test_line, "Helvetica", 9) < width - 100:
            line = test_line
        else:
            text_object.textLine(line)
            line = word + " "
    text_object.textLine(line)  # Add the last line
    
    c.drawText(text_object)
    
    # Save the PDF
    c.save()
    
    return output_path

# === Main Function ===
def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Load settings from arguments
    MODEL_PATH = args.model
    TEST_FOLDER = args.test_folder
    OUTPUT_FOLDER = args.output
    IMG_SIZE = args.img_size
    BATCH_MODE = args.batch
    
    # Define category mappings
    CATEGORIES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
    
    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Load model
    try:
        logger.info(f"Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Count total files to process
    total_files = 0
    for tumor_type in CATEGORIES:
        folder_path = os.path.join(TEST_FOLDER, tumor_type)
        if os.path.exists(folder_path):
            total_files += len([f for f in os.listdir(folder_path) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    logger.info(f"Found {total_files} images to process")
    
    # Confirm processing if not in batch mode
    if not BATCH_MODE and total_files > 0:
        confirm = input(f"Process {total_files} images? (y/n): ")
        if confirm.lower() != 'y':
            logger.info("Operation cancelled by user")
            return
    
    # Process images
    processed_count = 0
    with tqdm(total=total_files, desc="Processing Images") as pbar:
        for tumor_type in CATEGORIES:
            folder_path = os.path.join(TEST_FOLDER, tumor_type)
            if not os.path.exists(folder_path):
                continue
                
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder_path, filename)
                    
                    try:
                        # Read image
                        img = cv2.imread(image_path)
                        if img is None:
                            logger.warning(f"Failed to read image: {image_path}")
                            pbar.update(1)
                            continue
                        
                        # Preprocess
                        original, processed = preprocess(img, IMG_SIZE)
                        
                        # Predict
                        prediction = model.predict(processed, verbose=0)
                        label_idx = np.argmax(prediction)
                        label = CATEGORIES[label_idx]
                        confidence = float(np.max(prediction)) * 100
                        
                        # Check if no tumor was detected
                        if label == "no_tumor":
                            # For no tumor cases, create blank masks instead of segmenting
                            tumor_mask = np.zeros_like(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                            grown_mask = np.zeros_like(tumor_mask)
                            growth_percent = 0.0
                            risk_level = "None"
                            
                            # Create clean visualization without misleading tumor highlights
                            viz_paths = create_no_tumor_visualization(cv2.resize(original, (512, 512)))
                        else:
                            # Only segment tumor and simulate growth for actual tumor cases
                            tumor_mask = segment_tumor(img)
                            
                            # Simulate growth
                            grown_mask, growth_percent = simulate_growth(tumor_mask)
                            
                            # Determine risk level
                            if growth_percent > 30:
                                risk_level = "High"
                            elif growth_percent > 15:
                                risk_level = "Moderate"
                            else:
                                risk_level = "Low"
                            
                            # Create visualization for tumor cases
                            viz_paths = create_visualization(
                                cv2.resize(original, (512, 512)), 
                                cv2.resize(tumor_mask, (512, 512)),
                                cv2.resize(grown_mask, (512, 512))
                            )
                        
                        # Generate report with the correct risk level
                        output_pdf = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}_report.pdf")
                        generate_report(
                            image_path, label, confidence, 
                            growth_percent, viz_paths, output_pdf,
                            risk_level  # Pass the risk level as a parameter
                        )
                        
                        processed_count += 1
                        logger.debug(f"Processed {filename}: {label}, {confidence:.2f}%, Growth: {growth_percent:.1f}%")
                        pbar.update(1)
                        
                    except Exception as e:
                        logger.error(f"Error processing {filename}: {e}")
                        pbar.update(1)
                        continue
    
    # Clean up temporary files
    if os.path.exists("temp_visualization.png"):
        os.remove("temp_visualization.png")
    if os.path.exists("temp_visualization.jpg"):
        os.remove("temp_visualization.jpg")
    
    logger.info(f"✅ Successfully processed {processed_count} out of {total_files} images")
    logger.info(f"Reports saved to {os.path.abspath(OUTPUT_FOLDER)}")

if __name__ == "__main__":
    main()