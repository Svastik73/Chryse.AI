# 🌍 Chryse.AI

**Chryse.AI** is a Python-based toolkit designed for landscape delineation using multispectral and Synthetic Aperture Radar (SAR) imagery. It offers functionalities for:

- 🌈 Colorizing SAR images  
- 🌳 Deforestation detection  
- 🏔️ Avalanche delineation through an interactive Streamlit interface  

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/Svastik73/Chryse.AI.git
cd Chryse.AI
```
###Install Dependencies
```bash
pip install -r requirements.txt
```
## Usage
###1. SAR Image Colorization
To colorize SAR images using a Convolutional Neural Network (CNN):

```bash
python CNN_SAR_color.py
```
###2. Deforestation Detection
For detecting deforestation areas:

```bash
python deforestation_new.py
```
###3. Avalanche Delineation (Streamlit App)
Launch the Streamlit web application for interactive avalanche delineation:

```bash
streamlit run streamlit_avalanche.py
```
<h2>SAR image colorization</h2>
<p align="center">
  <img src="Images/ji.png" alt="UI" width="45%" />
    <img src="Images/jik.png" alt="UI" width="45%" />
</p>



<h2>Avalanche dilineation</H2>
<p align="center">
  <img src="Images/avalanche.1.png" alt="UI" width="45%" />
  <img src="Images/avalanche.png" alt="Output" width="45%" />
</p>
<h2>Geographic loss </h2>

<img src="Images/single_image_deforestation_results.png" alt="deforest" width="60%">
