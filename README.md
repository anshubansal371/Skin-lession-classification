# 🌟 Skin Lesion Classification Using SVM  

A Machine Learning project that classifies skin lesion images into different disease categories using a Support Vector Machine (SVM).  
This project demonstrates practical ML workflow, ideal for students and beginners applying for internships.

---

## 📌 Project Overview  
Skin diseases are among the most common health issues globally. Early and accurate detection can help improve treatment outcomes.  
This project uses **image preprocessing**, **feature extraction**, and an **SVM classifier** to predict different types of skin lesions.

---

## 🧠 Model Used  
### **Support Vector Machine (SVM)**
- Works well for high-dimensional image features  
- Performs reliably on medium-sized datasets  
- Achieved **70%–80% accuracy** on a dataset of ~10,000 images  

---

## 📊 Dataset  
- **Total Images:** ~10,000  
- **Image Type:** Skin lesion / disease images  
- **Classes:** Multiple disease categories  
- **Note:**  
  The dataset is **not included in this repository** due to large size.  
  You can place your dataset inside a `data/` folder as shown below.

---

## 🗂 Project Structure  
Skin-Lesion-Classification/
│
├── data/ # Dataset (ignored by Git)
├── models/ # Saved ML models (.pkl)
├── src/
│ ├── preprocessing.py
│ ├── training.py
│ ├── inference.py
│ └── utils.py
│
├── requirements.txt
├── README.md
└── app.py

## 🚀 Future Improvements  
- Replace SVM with **CNN / Transfer Learning (ResNet, MobileNet, EfficientNet)**  
- Add image augmentation  
- Train on larger dermatology datasets  
- Deploy model on cloud  
