# 🔊 Empowering Voice Biometrics: A Comprehensive Approach with the XEUS Model  

## 📌 Overview  
This repository contains the implementation and methodology for a **text-independent speaker verification system** using the **XEUS model** from the ESPnet framework. The project leverages **self-supervised learning (SSL)** techniques to analyze multilingual voice samples and extract speaker embeddings for authentication.  

## 🎯 Objective  
- Develop a robust **voice biometric system** for speaker recognition across multiple languages.  
- Utilize **phonetic and prosodic features** for improved accuracy.  
- Implement a **threshold-based anomaly detection algorithm** to verify speakers.  

## 📖 Methodology  

1. **Dataset Preparation**  
   - Voice samples sourced from **VoxCeleb1 Dataset (Indian celebrities)**.  
   - Organized into **folders with unique speaker IDs** for enrollment and testing.  

2. **Feature Extraction**  
   - XEUS model extracts **speaker embeddings** from raw audio without spectrogram conversion.  
   - Embeddings capture key voice attributes like **pitch, tone, and accent**.  

3. **Speaker Enrollment**  
   - Multiple audio samples per speaker are averaged to create a **robust voice representation**.  

4. **Speaker Verification**  
   - A test voice sample is compared with stored embeddings using **Euclidean distance**.  
   - A predefined **threshold** determines whether the match is valid.  

## 🛠 Tech Stack  
- **Python** for implementation  
- **ESPnet & XEUS Model** for voice processing  
- **NumPy, SciPy** for distance calculations  
- **Pandas, Matplotlib** for data analysis and visualization  

## 📂 Repository Structure  
```
📂 Empowering-Voice-Biometrics  
 ├── 📄 README.md                # Project documentation  
 ├── 📄 requirements.txt         # Dependencies  
 ├── 📂 data/                    # Audio datasets  
 ├── 📂 src/                     # Source code  
 │   ├── enroll.py               # Speaker enrollment  
 │   ├── verify.py               # Speaker verification  
 │   ├── feature_extraction.py    # XEUS embeddings  
 │   ├── utils.py                 # Helper functions  
```

## 🔬 Experimental Results  
- High **recognition accuracy** in text-independent authentication.  
- **Multilingual support** enhances inclusivity in voice biometrics.  

## 🚀 How to Use  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/Empowering-Voice-Biometrics.git
   cd Empowering-Voice-Biometrics
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

## 🔗 References  
- [VoxCeleb1 Dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)  
- [ESPnet Framework](https://espnet.github.io/)  
- [XEUS Model Paper](https://arxiv.org/abs/XXXX.XXXX)  

## 🤝 Contributors  
- **Gnanamoorthi PV**  
- **M Balasubramanian**  
- **Harshini KN**  
- **Chitra P**  

