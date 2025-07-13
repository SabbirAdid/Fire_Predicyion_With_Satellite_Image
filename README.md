# Fire_Predicyion_With_Satellite_Image
Detect, analyze, and predict wildfire occurrences and spread in real-time

# Fire Prediction with Satellite Images 🌍🔥

This project uses **satellite imagery** and **deep learning** techniques to predict the occurrence of wildfires. The model uses the **MobileNetV2** architecture to classify satellite images into two categories: **Wildfire** and **Non-Wildfire**.

## 🚀 Project Overview

The aim of this project is to create a machine learning pipeline that automatically classifies satellite images to predict wildfire occurrences. This can be useful in monitoring and responding to wildfires in real-time. The project utilizes the following steps:

- **Data Preprocessing**: Extracting and cleaning image data.
- **Image Processing**: Resizing and normalizing images for model training.
- **Model Training**: Fine-tuning the MobileNetV2 model for classification.
- **Dataset Balancing**: Ensuring the dataset is balanced for fair predictions.
- **Model Evaluation**: Evaluating the model's performance and generating results.

## 📦 Installation and Setup

Follow these steps to set up the project on your local machine.

### 1. Install Dependencies

Install the necessary libraries using `pip`. Open a terminal and run the following command:

```bash
pip install tensorflow opencv-python matplotlib scikit-learn tqdm
```

### 2. Prepare Your Dataset

Upload your dataset in `.zip` format. This should contain images categorized into **Wildfire** and **Non-Wildfire** folders.

### 3. Extract Dataset

Once the dataset is uploaded, it will be automatically extracted into the appropriate directory.

## 📝 Steps in the Project

### Step 1: Import Libraries

In this step, we import the necessary libraries for processing images, building the model, and evaluating it.

```python
import os, zipfile, cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
```

### Step 2: Dataset Extraction

The dataset will be uploaded and extracted to a specified directory. This is where the images will be organized for further processing.

```python
uploaded = files.upload()
zip_filename = next(iter(uploaded))
extract_path = "/content/dataset_split"
with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
```

### Step 3: Image Processing

Images are processed by resizing them to a uniform size (350x350) and normalizing pixel values to range between 0 and 1. 

```python
def process_images(folder_path, label):
    if not folder_path or not os.path.exists(folder_path):
        print(f"🚫 Folder not found: {folder_path}")
        return
    files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[-1].lower() in valid_extensions]
    valid = 0
    for img_name in tqdm(files, desc=f"Processing {label}"):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize image
        img = img / 255.0  # Normalize image
        data.append((img_path, label))  # Append processed data
        valid += 1
    print(f"✅ Valid images: {valid}/{len(files)}")
```

### Step 4: Dataset Balancing

In this step, the dataset is balanced by ensuring that both categories (wildfire and non-wildfire) have an equal number of images. This prevents model bias towards the more frequent class.

```python
balanced_data = wildfire_data[:min_len] + nonwildfire_data[:min_len]
random.shuffle(balanced_data)
df = pd.DataFrame(balanced_data, columns=["image_path", "label"])
csv_path = "/content/fire_labels.csv"
df.to_csv(csv_path, index=False)
files.download(csv_path)
```

### Step 5: Model Training

The model is built using **MobileNetV2** as the base model and fine-tuned on the processed images. The model is compiled with the **Adam optimizer** and trained using callbacks to reduce the learning rate when the model stops improving.

```python
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
```

### Step 6: Model Evaluation & Results

After training, the model is evaluated, and a CSV file containing image paths and corresponding labels is generated.

```python
model.evaluate(validation_data)
```

## 🧑‍💻 Usage

### 1. Upload Your Data

Simply upload your dataset in `.zip` format.

### 2. Run the Notebook

After setting up the environment and uploading your data, execute the code in the notebook to train the model.

### 3. Evaluate Model Performance

After training, evaluate the model’s performance using the validation set.

## 🔧 Additional Features

- **Early Stopping**: Stop training when the model stops improving to save time.
- **Learning Rate Scheduler**: Reduce the learning rate when the model performance plateaus.

## 🚧 Future Improvements

- Experiment with other neural network architectures (e.g., ResNet, Inception).
- Improve preprocessing techniques (e.g., data augmentation).
- Integrate real-time satellite imagery for prediction.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

---

Feel free to explore the notebook and improve it further. Contributions are welcome! ✨
