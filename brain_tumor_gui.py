import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
from tensorflow.keras.models import load_model
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from datetime import datetime
import logging
import glob


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("tumor_detection_gui.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Define categories
CATEGORIES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]


class BrainTumorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Tumor Detection System")
        self.root.geometry("1200x750")
        self.root.configure(bg="#f0f0f0")

        # Theme colors
        self.primary_color = "#3498db"
        self.secondary_color = "#2980b9"
        self.bg_color = "#f0f0f0"
        self.text_color = "#333333"

        # Variables
        self.model = None
        self.model_path = tk.StringVar(value="Brain_Tumor_Detection_Model.keras")
        self.output_folder = tk.StringVar(value="AI_Reports")
        self.current_image_path = None
        self.img_size = 128
        self.photo = None

        # Styles
        self.style = ttk.Style()
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure(
            "TButton",
            background=self.primary_color,
            foreground="white",
            font=("Arial", 10, "bold"),
            padding=5
        )
        self.style.map("TButton", background=[("active", self.secondary_color)])
        self.style.configure(
            "TLabel",
            background=self.bg_color,
            foreground=self.text_color,
            font=("Arial", 10)
        )
        self.style.configure(
            "Header.TLabel",
            background=self.bg_color,
            foreground=self.primary_color,
            font=("Arial", 14, "bold")
        )

        # Build UI
        self.create_frames()
        self.create_sidebar()
        self.create_image_area()
        self.create_results_area()

        # Auto load model
        self.load_model_from_path()

    def create_frames(self):
        self.sidebar_frame = ttk.Frame(self.root, width=250, style="TFrame")
        self.sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.main_frame = ttk.Frame(self.root, style="TFrame")
        self.main_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.image_frame = ttk.Frame(self.main_frame, style="TFrame")
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.results_frame = ttk.Frame(self.main_frame, style="TFrame")
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_sidebar(self):
        style = ttk.Style()
        style.configure("Black.TButton", foreground="black")

        title_label = ttk.Label(
            self.sidebar_frame,
            text="Brain Tumor\nDetection System",
            style="Header.TLabel",
            anchor="center"
        )
        title_label.pack(pady=20, fill=tk.X)

        # Model section
        model_frame = ttk.Frame(self.sidebar_frame, style="TFrame")
        model_frame.pack(fill=tk.X, pady=10)

        ttk.Label(model_frame, text="Model Path:", style="TLabel").pack(anchor=tk.W)

        model_entry = ttk.Entry(model_frame, textvariable=self.model_path, width=25)
        model_entry.pack(fill=tk.X, pady=5)

        ttk.Button(
            model_frame,
            text="Browse Model",
            command=self.browse_model,
            style="Black.TButton"
        ).pack(fill=tk.X)

        ttk.Button(
            model_frame,
            text="Load Model",
            command=self.load_model_from_path,
            style="Black.TButton"
        ).pack(fill=tk.X, pady=5)

        # Output section
        output_frame = ttk.Frame(self.sidebar_frame, style="TFrame")
        output_frame.pack(fill=tk.X, pady=10)

        ttk.Label(output_frame, text="Output Folder:", style="TLabel").pack(anchor=tk.W)

        output_entry = ttk.Entry(output_frame, textvariable=self.output_folder, width=25)
        output_entry.pack(fill=tk.X, pady=5)

        ttk.Button(
            output_frame,
            text="Browse Output",
            command=self.browse_output,
            style="Black.TButton"
        ).pack(fill=tk.X)

        # Action buttons
        action_frame = ttk.Frame(self.sidebar_frame, style="TFrame")
        action_frame.pack(fill=tk.X, pady=20)

        ttk.Button(
            action_frame,
            text="Select Image",
            command=self.browse_image,
            style="Black.TButton"
        ).pack(fill=tk.X, pady=5)

        ttk.Button(
            action_frame,
            text="Process Image",
            command=self.process_current_image,
            style="Black.TButton"
        ).pack(fill=tk.X, pady=5)

        ttk.Button(
            action_frame,
            text="Batch Process",
            command=self.batch_process,
            style="Black.TButton"
        ).pack(fill=tk.X, pady=5)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(
            self.sidebar_frame,
            textvariable=self.status_var,
            style="TLabel"
        ).pack(side=tk.BOTTOM, pady=10)

    def create_image_area(self):
        ttk.Label(
            self.image_frame,
            text="MRI Image",
            style="Header.TLabel"
        ).pack(pady=5)

        self.image_canvas = tk.Canvas(
            self.image_frame,
            bg="white",
            width=350,
            height=350
        )
        self.image_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.display_placeholder()

    def create_results_area(self):
        ttk.Label(
            self.results_frame,
            text="Analysis Results",
            style="Header.TLabel"
        ).pack(pady=5)

        results_display = ttk.Frame(self.results_frame, style="TFrame")
        results_display.pack(fill=tk.BOTH, expand=True)

        self.text_results_frame = ttk.Frame(results_display, style="TFrame")
        self.text_results_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        fields_frame = ttk.Frame(self.text_results_frame, style="TFrame")
        fields_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tumor type
        ttk.Label(
            fields_frame,
            text="Tumor Type:",
            style="TLabel",
            font=("Arial", 11, "bold")
        ).grid(row=0, column=0, sticky=tk.W, pady=5)

        self.diagnosis_var = tk.StringVar(value="N/A")
        ttk.Label(
            fields_frame,
            textvariable=self.diagnosis_var,
            style="TLabel",
            font=("Arial", 11)
        ).grid(row=0, column=1, sticky=tk.W, pady=5)

        # Confidence
        ttk.Label(
            fields_frame,
            text="Confidence:",
            style="TLabel",
            font=("Arial", 11, "bold")
        ).grid(row=1, column=0, sticky=tk.W, pady=5)

        self.confidence_var = tk.StringVar(value="N/A")
        ttk.Label(
            fields_frame,
            textvariable=self.confidence_var,
            style="TLabel",
            font=("Arial", 11)
        ).grid(row=1, column=1, sticky=tk.W, pady=5)

        # Growth projection
        ttk.Label(
            fields_frame,
            text="Growth Projection:",
            style="TLabel",
            font=("Arial", 11, "bold")
        ).grid(row=2, column=0, sticky=tk.W, pady=5)

        self.growth_var = tk.StringVar(value="N/A")
        ttk.Label(
            fields_frame,
            textvariable=self.growth_var,
            style="TLabel",
            font=("Arial", 11)
        ).grid(row=2, column=1, sticky=tk.W, pady=5)

        # Risk
        ttk.Label(
            fields_frame,
            text="Risk Level:",
            style="TLabel",
            font=("Arial", 11, "bold")
        ).grid(row=3, column=0, sticky=tk.W, pady=5)

        self.risk_var = tk.StringVar(value="N/A")
        ttk.Label(
            fields_frame,
            textvariable=self.risk_var,
            style="TLabel",
            font=("Arial", 11)
        ).grid(row=3, column=1, sticky=tk.W, pady=5)

        # Report
        ttk.Label(
            fields_frame,
            text="Report:",
            style="TLabel",
            font=("Arial", 11, "bold")
        ).grid(row=4, column=0, sticky=tk.W, pady=5)

        self.report_var = tk.StringVar(value="Not generated")
        ttk.Label(
            fields_frame,
            textvariable=self.report_var,
            style="TLabel",
            font=("Arial", 11)
        ).grid(row=4, column=1, sticky=tk.W, pady=5)

        report_btn_frame = ttk.Frame(fields_frame, style="TFrame")
        report_btn_frame.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=10)

        ttk.Button(
            report_btn_frame,
            text="Generate Report",
            command=self.generate_report
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            report_btn_frame,
            text="View Report",
            command=self.view_report
        ).pack(side=tk.LEFT, padx=5)

        self.viz_frame = ttk.Frame(results_display, style="TFrame")
        self.viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.viz_placeholder = ttk.Label(
            self.viz_frame,
            text="Tumor visualization will appear here",
            style="TLabel"
        )
        self.viz_placeholder.pack(expand=True)

    def display_placeholder(self):
        self.image_canvas.delete("all")
        self.image_canvas.create_text(
            250,
            250,
            text="Select an MRI image to analyze",
            font=("Arial", 14),
            fill=self.text_color
        )

    def browse_model(self):
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Keras Model", "*.keras"), ("H5 Model", "*.h5"), ("All Files", "*.*")]
        )
        if model_path:
            self.model_path.set(model_path)

    def browse_output(self):
        output_folder = filedialog.askdirectory(title="Select Output Folder")
        if output_folder:
            self.output_folder.set(output_folder)
            os.makedirs(output_folder, exist_ok=True)

    def browse_image(self):
        image_path = filedialog.askopenfilename(
            title="Select MRI Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg"), ("All Files", "*.*")]
        )
        if image_path:
            self.current_image_path = image_path
            self.display_image(image_path)
            self.reset_results()

    def load_model_from_path(self):
        try:
            model_path = self.model_path.get()
            if not os.path.exists(model_path):
                messagebox.showwarning("Warning", f"Model file not found: {model_path}")
                self.status_var.set("Model not loaded. File not found.")
                return False

            tf.keras.backend.clear_session()

            self.status_var.set("Loading model...")
            self.root.update()

            self.model = load_model(model_path)

            self.status_var.set(f"Model loaded: {os.path.basename(model_path)}")
            return True

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            logger.error(f"Failed to load model: {e}")
            self.status_var.set("Model loading failed.")
            return False

    def display_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not read selected image.")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            canvas_width = max(self.image_canvas.winfo_width(), 350)
            canvas_height = max(self.image_canvas.winfo_height(), 350)

            h, w = img.shape[:2]
            if h > w:
                new_h = min(h, canvas_height)
                new_w = int(w * new_h / h)
            else:
                new_w = min(w, canvas_width)
                new_h = int(h * new_w / w)

            if new_w > 0 and new_h > 0:
                img = cv2.resize(img, (new_w, new_h))

            img = Image.fromarray(img)
            self.photo = ImageTk.PhotoImage(image=img)

            self.image_canvas.delete("all")
            x = (canvas_width - self.photo.width()) // 2
            y = (canvas_height - self.photo.height()) // 2
            self.image_canvas.create_image(x, y, anchor=tk.NW, image=self.photo)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {e}")
            logger.error(f"Failed to display image: {e}")
            self.display_placeholder()

    def reset_results(self):
        self.diagnosis_var.set("N/A")
        self.confidence_var.set("N/A")
        self.growth_var.set("N/A")
        self.risk_var.set("N/A")
        self.report_var.set("Not generated")

        for widget in self.viz_frame.winfo_children():
            widget.destroy()

        self.viz_placeholder = ttk.Label(
            self.viz_frame,
            text="Tumor visualization will appear here",
            style="TLabel"
        )
        self.viz_placeholder.pack(expand=True)

    def preprocess(self, img, img_size):
        original = img.copy()
        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0
        return original, np.expand_dims(img, axis=0)

    def segment_tumor(self, img):
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        thresh = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        contours, _ = cv2.findContours(sure_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros_like(gray)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < 10000:
                cv2.drawContours(mask, [cnt], 0, 255, -1)

        return mask

    def simulate_growth(self, mask, growth_factor=1.5):
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        initial_area = np.count_nonzero(binary)

        kernel_size = max(1, int(3 * growth_factor))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        grown_mask = cv2.dilate(binary, kernel, iterations=2)

        grown_area = np.count_nonzero(grown_mask)

        if initial_area > 0:
            growth_percent = ((grown_area - initial_area) / initial_area) * 100
        else:
            growth_percent = 0

        return grown_mask, growth_percent

    def create_visualization(self, original, tumor_mask, grown_mask, label):
        for widget in self.viz_frame.winfo_children():
            widget.destroy()

        fig = plt.Figure(figsize=(8, 3), dpi=100)

        ax1 = fig.add_subplot(131)
        ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        ax1.set_title("Original MRI")
        ax1.axis("off")

        ax2 = fig.add_subplot(132)
        ax2.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        if "no_tumor" not in label.lower():
            h, w = original.shape[:2]
            resized_tumor = cv2.resize(tumor_mask, (w, h))
            mask_overlay = np.zeros_like(original)
            mask_overlay[:, :, 1] = resized_tumor
            ax2.imshow(mask_overlay, alpha=0.4)
            ax2.set_title("Day 1 - Detected Tumor")
        else:
            ax2.set_title("No Tumor Detected")
        ax2.axis("off")

        ax3 = fig.add_subplot(133)
        ax3.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        if "no_tumor" not in label.lower():
            h, w = original.shape[:2]
            resized_grown = cv2.resize(grown_mask, (w, h))
            grown_overlay = np.zeros_like(original)
            grown_overlay[:, :, 0] = resized_grown
            ax3.imshow(grown_overlay, alpha=0.4)
            ax3.set_title("Day 30 - Projected Growth")
        else:
            ax3.set_title("Healthy Brain")
        ax3.axis("off")

        fig.tight_layout()

        canvas_widget = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def process_current_image(self):
        if not self.current_image_path:
            messagebox.showinfo("Info", "Please select an image first.")
            return

        if not self.model:
            if not self.load_model_from_path():
                return

        self.status_var.set("Processing image...")
        self.root.update()

        try:
            img = cv2.imread(self.current_image_path)
            if img is None:
                raise ValueError(f"Failed to read image: {self.current_image_path}")

            original, processed = self.preprocess(img, self.img_size)

            prediction = self.model.predict(processed, verbose=0)
            label_idx = np.argmax(prediction)
            label = CATEGORIES[label_idx]
            confidence = float(np.max(prediction)) * 100

            if label == "glioma_tumor":
                tumor_type = "Glioma"
            elif label == "meningioma_tumor":
                tumor_type = "Meningioma"
            elif label == "pituitary_tumor":
                tumor_type = "Pituitary"
            else:
                tumor_type = "No Tumor"

            if label == "no_tumor":
                tumor_mask = np.zeros_like(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                grown_mask = np.zeros_like(tumor_mask)
                growth_percent = 0.0
                risk_level = "None"
            else:
                tumor_mask = self.segment_tumor(img)
                grown_mask, growth_percent = self.simulate_growth(tumor_mask)

                if growth_percent > 30:
                    risk_level = "High"
                elif growth_percent > 15:
                    risk_level = "Moderate"
                else:
                    risk_level = "Low"

            self.diagnosis_var.set(tumor_type)
            self.confidence_var.set(f"{confidence:.2f}%")
            self.growth_var.set(f"{growth_percent:.1f}%")
            self.risk_var.set(risk_level)
            self.report_var.set("Ready to generate")

            self.create_visualization(original, tumor_mask, grown_mask, label)

            self.status_var.set("Processing complete")

        except Exception as e:
            messagebox.showerror("Error", f"Error processing image: {e}")
            logger.error(f"Error processing image: {e}")
            self.status_var.set("Processing failed")

    def generate_report(self):
        if not self.current_image_path or self.diagnosis_var.get() == "N/A":
            messagebox.showinfo("Info", "Please process an image first.")
            return

        try:
            output_dir = self.output_folder.get()
            os.makedirs(output_dir, exist_ok=True)

            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            output_pdf = os.path.join(output_dir, f"{base_name}_report.pdf")

            self.status_var.set("Generating report...")
            self.root.update()

            threading.Thread(
                target=self._generate_report_thread,
                args=(output_pdf,),
                daemon=True
            ).start()

        except Exception as e:
            messagebox.showerror("Error", f"Error generating report: {e}")
            logger.error(f"Error generating report: {e}")
            self.status_var.set("Report generation failed")

    def _generate_report_thread(self, output_pdf):
        try:
            label = self.diagnosis_var.get().lower().replace(" ", "_")
            confidence = float(self.confidence_var.get().strip("%"))
            growth_percent = float(self.growth_var.get().strip("%"))
            risk_level = self.risk_var.get()

            img = cv2.imread(self.current_image_path)
            original, _ = self.preprocess(img, self.img_size)

            if "no_tumor" in label:
                tumor_mask = np.zeros_like(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                grown_mask = np.zeros_like(tumor_mask)
            else:
                tumor_mask = self.segment_tumor(img)
                grown_mask, _ = self.simulate_growth(tumor_mask)

            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            plt.title("Original MRI")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            if "no_tumor" not in label:
                mask_overlay = np.zeros_like(original)
                mask_overlay[:, :, 1] = cv2.resize(tumor_mask, (original.shape[1], original.shape[0]))
                plt.imshow(mask_overlay, alpha=0.4)
                plt.title("Day 1 - Detected Tumor")
            else:
                plt.title("No Tumor Detected")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            if "no_tumor" not in label:
                grown_overlay = np.zeros_like(original)
                grown_overlay[:, :, 0] = cv2.resize(grown_mask, (original.shape[1], original.shape[0]))
                plt.imshow(grown_overlay, alpha=0.4)
                plt.title("Day 30 - Projected Growth")
            else:
                plt.title("Healthy Brain")
            plt.axis("off")

            viz_path_png = "temp_visualization.png"
            viz_path_jpg = "temp_visualization.jpg"
            plt.savefig(viz_path_png, dpi=150, bbox_inches="tight")
            plt.savefig(viz_path_jpg, dpi=150, bbox_inches="tight")
            plt.close()

            self._create_pdf_report(
                output_pdf,
                label,
                confidence,
                growth_percent,
                risk_level,
                (viz_path_png, viz_path_jpg)
            )

            self.root.after(0, self._report_completion, output_pdf)

        except Exception as e:
            logger.error(f"Error in report generation thread: {e}")
            self.root.after(0, self._report_error, str(e))

    def _create_pdf_report(self, output_path, label, confidence, growth_percent, risk_level, viz_img_paths):
        c = canvas.Canvas(output_path, pagesize=letter)
        width, height = letter

        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(width / 2, height - 50, "Brain Tumor Analysis Report")

        c.setStrokeColor(colors.grey)
        c.line(50, height - 70, width - 50, height - 70)

        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 100, "Patient Information")
        c.setFont("Helvetica", 12)
        patient_id = os.path.splitext(os.path.basename(self.current_image_path))[0]
        c.drawString(50, height - 120, f"Patient ID: {patient_id}")
        c.drawString(50, height - 140, f"Scan Date: {datetime.now().strftime('%Y-%m-%d')}")
        c.drawString(50, height - 160, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 200, "AI Diagnosis")
        c.setFont("Helvetica", 12)

        if "no_tumor" in label.lower():
            c.setFillColor(colors.green)
        else:
            c.setFillColor(colors.red)

        c.drawString(50, height - 220, f"Diagnosis: {label.replace('_', ' ').title()}")
        c.setFillColor(colors.black)

        c.drawString(50, height - 240, f"Model Confidence: {confidence:.2f}%")

        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 280, "Tumor Growth Projection")
        c.setFont("Helvetica", 12)

        if "no_tumor" in label.lower():
            c.drawString(50, height - 300, "Estimated 30-Day Growth: N/A (No tumor detected)")
            c.drawString(50, height - 320, f"Growth Risk Level: {risk_level}")
        else:
            c.drawString(50, height - 300, f"Estimated 30-Day Growth: {growth_percent:.1f}%")
            c.drawString(50, height - 320, f"Growth Risk Level: {risk_level}")

        viz_img_path_png, viz_img_path_jpg = viz_img_paths
        img_width = width - 100
        img_height = img_width / 3

        try:
            viz_img = ImageReader(viz_img_path_png)
            c.drawImage(viz_img, 50, height - 540, width=img_width, height=img_height)
        except Exception as e:
            logger.warning(f"Failed to read PNG visualization: {e}")
            try:
                viz_img = ImageReader(viz_img_path_jpg)
                c.drawImage(viz_img, 50, height - 540, width=img_width, height=img_height)
            except Exception as e2:
                logger.warning(f"Failed to read JPG visualization: {e2}")

        c.setFont("Helvetica", 9)
        disclaimer = (
            "DISCLAIMER: This is an AI-generated report for research purposes only. "
            "The growth simulation is based on computer algorithms and should not be used "
            "for clinical decision-making. Always consult with a qualified medical "
            "professional for proper diagnosis and treatment."
        )

        text_object = c.beginText(50, 50)
        text_object.setFont("Helvetica", 9)

        words = disclaimer.split()
        line = ""
        for word in words:
            test_line = line + word + " "
            if c.stringWidth(test_line, "Helvetica", 9) < width - 100:
                line = test_line
            else:
                text_object.textLine(line)
                line = word + " "
        text_object.textLine(line)

        c.drawText(text_object)
        c.save()

        try:
            if os.path.exists(viz_img_path_png):
                os.remove(viz_img_path_png)
            if os.path.exists(viz_img_path_jpg):
                os.remove(viz_img_path_jpg)
        except Exception as e:
            logger.warning(f"Error cleaning up temp files: {e}")

        return output_path

    def _report_completion(self, output_path):
        self.report_var.set(os.path.basename(output_path))
        self.status_var.set("Report generated successfully")
        messagebox.showinfo("Success", f"Report saved to: {output_path}")

    def _report_error(self, error_msg):
        self.status_var.set("Report generation failed")
        messagebox.showerror("Error", f"Failed to generate report: {error_msg}")

    def view_report(self):
        if self.report_var.get() == "Not generated":
            messagebox.showinfo("Info", "Please generate a report first.")
            return

        output_dir = self.output_folder.get()
        report_name = self.report_var.get()
        report_path = os.path.join(output_dir, report_name)

        if not os.path.exists(report_path):
            messagebox.showinfo("Info", f"Report file not found: {report_path}")
            return

        try:
            import platform
            import subprocess

            if platform.system() == "Darwin":
                subprocess.run(["open", report_path])
            elif platform.system() == "Windows":
                os.startfile(report_path)
            else:
                subprocess.run(["xdg-open", report_path])

        except Exception as e:
            messagebox.showerror("Error", f"Failed to open report: {e}")
            logger.error(f"Failed to open report: {e}")

    def batch_process(self):
        if not self.model:
            if not self.load_model_from_path():
                return

        folder_path = filedialog.askdirectory(title="Select Folder with MRI Images")
        if not folder_path:
            return

        image_files = []
        for ext in [".png", ".jpg", ".jpeg"]:
            image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))

        if not image_files:
            messagebox.showinfo("Info", "No image files found in the selected folder.")
            return

        if not messagebox.askyesno(
            "Batch Processing",
            f"Process {len(image_files)} images?\nThis may take some time."
        ):
            return

        threading.Thread(
            target=self._batch_process_thread,
            args=(image_files,),
            daemon=True
        ).start()

    def _batch_process_thread(self, image_files):
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Batch Processing")
        progress_window.geometry("400x150")
        progress_window.transient(self.root)
        progress_window.grab_set()

        ttk.Label(progress_window, text="Processing images...").pack(pady=10)

        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=len(image_files))
        progress_bar.pack(fill=tk.X, padx=20, pady=10)

        status_var = tk.StringVar(value="Starting...")
        ttk.Label(progress_window, textvariable=status_var).pack(pady=10)

        success_count = 0

        for i, image_path in enumerate(image_files):
            viz_path_png = None
            viz_path_jpg = None

            try:
                progress_var.set(i)
                status_var.set(f"Processing {os.path.basename(image_path)}...")
                progress_window.update()

                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"Failed to read image: {image_path}")

                original, processed = self.preprocess(img, self.img_size)

                prediction = self.model.predict(processed, verbose=0)
                label_idx = np.argmax(prediction)
                label = CATEGORIES[label_idx]
                confidence = float(np.max(prediction)) * 100

                if label == "no_tumor":
                    tumor_mask = np.zeros_like(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                    grown_mask = np.zeros_like(tumor_mask)
                    growth_percent = 0.0
                    risk_level = "None"
                else:
                    tumor_mask = self.segment_tumor(img)
                    grown_mask, growth_percent = self.simulate_growth(tumor_mask)

                    if growth_percent > 30:
                        risk_level = "High"
                    elif growth_percent > 15:
                        risk_level = "Moderate"
                    else:
                        risk_level = "Low"

                plt.figure(figsize=(15, 5))

                plt.subplot(1, 3, 1)
                plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
                plt.title("Original MRI")
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
                if "no_tumor" not in label.lower():
                    mask_overlay = np.zeros_like(original)
                    mask_overlay[:, :, 1] = cv2.resize(tumor_mask, (original.shape[1], original.shape[0]))
                    plt.imshow(mask_overlay, alpha=0.4)
                    plt.title("Day 1 - Detected Tumor")
                else:
                    plt.title("No Tumor Detected")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
                if "no_tumor" not in label.lower():
                    grown_overlay = np.zeros_like(original)
                    grown_overlay[:, :, 0] = cv2.resize(grown_mask, (original.shape[1], original.shape[0]))
                    plt.imshow(grown_overlay, alpha=0.4)
                    plt.title("Day 30 - Projected Growth")
                else:
                    plt.title("Healthy Brain")
                plt.axis("off")

                viz_path_png = f"temp_viz_{i}.png"
                viz_path_jpg = f"temp_viz_{i}.jpg"
                plt.savefig(viz_path_png, dpi=150, bbox_inches="tight")
                plt.savefig(viz_path_jpg, dpi=150, bbox_inches="tight")
                plt.close()

                output_dir = self.output_folder.get()
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_pdf = os.path.join(output_dir, f"{base_name}_report.pdf")

                self._create_pdf_report(
                    output_pdf,
                    label,
                    confidence,
                    growth_percent,
                    risk_level,
                    (viz_path_png, viz_path_jpg)
                )

                success_count += 1

            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")

            finally:
                for temp_file in [viz_path_png, viz_path_jpg]:
                    if temp_file and os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except Exception:
                            pass

        progress_var.set(len(image_files))
        status_var.set(f"Finished: {success_count}/{len(image_files)} successful")

        progress_window.after(2000, progress_window.destroy)

        self.root.after(
            0,
            lambda: self.status_var.set(
                f"Batch processing complete: {success_count}/{len(image_files)}"
            )
        )

        self.root.after(
            0,
            lambda: messagebox.showinfo(
                "Batch Complete",
                f"Processed {len(image_files)} images.\n"
                f"Success: {success_count}\n"
                f"Failed: {len(image_files) - success_count}\n\n"
                f"Reports saved to: {os.path.abspath(self.output_folder.get())}"
            )
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = BrainTumorApp(root)
    root.mainloop()