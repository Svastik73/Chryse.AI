🌍 Chryse.AI
Chryse.AI is a Python-based toolkit designed for landscape delineation using multispectral and Synthetic Aperture Radar (SAR) imagery. It offers functionalities for:

Colorizing SAR images

Deforestation detection

Avalanche delineation through an interactive Streamlit interface
YouTube
+1
GitHub
+1

📂 Project Structure
bash
Copy
Edit

Chryse.AI/
├── CNN_SAR_color.py           # SAR image colorization
├── deforestation_new.py       # Deforestation detection
├── streamlit_avalanche.py     # Streamlit app for avalanche delineation
├── requirements.txt           # Python dependencies
├── Images/                    # Sample input/output images
└── README.md
⚙️ Installation
Clone the Repository

bash
Copy
Edit
git clone https://github.com/Svastik73/Chryse.AI.git
cd Chryse.AI
(Optional) Create and Activate a Virtual Environment

For macOS/Linux:

bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate
For Windows:

bash
Copy
Edit
python -m venv venv
venv\Scripts\activate
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
🚀 Usage
1. SAR Image Colorization
To colorize SAR images using a Convolutional Neural Network (CNN):

bash
Copy
Edit
python CNN_SAR_color.py
2. Deforestation Detection
For detecting deforestation areas:

bash
Copy
Edit
python deforestation_new.py
3. Avalanche Delineation (Streamlit App)
Launch the Streamlit web application for interactive avalanche delineation:

bash
Copy
Edit
streamlit run streamlit_avalanche.py
This will open a web interface in your default browser.

🖼️ Sample Outputs
SAR Image Colorization
YouTube
+1
GitHub
+1

Deforestation Detection

Avalanche Delineation Interface
YouTube
+3
GitHub
+3
GitHub
+3

Note: Replace the image paths with actual images from the Images/ directory.

🧩 Dependencies
Ensure the following Python packages are installed (as specified in requirements.txt):

numpy

pandas

matplotlib

opencv-python

streamlit

scikit-learn

tensorflow

keras

rasterio

gdal
Build5Nines
+4
GitHub
+4
YouTube
+4
Medium

Note: Some packages may require additional system dependencies, especially for gdal. Ensure they are installed accordingly.

🤝 Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to fork the repository and submit a pull request.

📄 License
This project is licensed under the MIT License.

📬 Contact
For questions or collaborations, please open an issue on the GitHub repository.




<h2>SAR image colorization</h2>
<p align="center">
  <img src="Images/ji.png" alt="UI" width="45%" />
  
</p>



<h2>Avalanche dilineation</H2>
<p align="center">
  <img src="Images/avalanche.1.png" alt="UI" width="45%" />
  <img src="Images/avalanche.png" alt="Output" width="45%" />
</p>
<h2>Geographic loss </h2>

<img src="Images/single_image_deforestation_results.png" alt="deforest" width="60%">
