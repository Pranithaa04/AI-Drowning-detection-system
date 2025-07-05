import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from collections import Counter
import time
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import joblib

img_size = (64,64)
X, y = [], []
model_accuracies = {}
data_path = "./classification_data"

for class_folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, class_folder)
    print(f"\nClass: {class_folder}")
    if not os.path.isdir(folder_path):
        continue

    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)

        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                img = load_img(img_path, target_size=img_size)
                img_array = img_to_array(img) / 255.0
                X.append(img_array)
                y.append(class_folder)
                print(f"✅ Loaded: {img_file}")
            except Exception as e:
                print(f"❌ Error loading {img_file}: {e}")
        else:
            print(f"⚠️ Skipped (not image): {img_file}")


print("Original class distribution:", Counter(y))

X = np.array(X)
print("X shape before flattening:", X.shape)

# Flatten the image arrays for ML models
X_flat = X.reshape(len(X), -1)
X_train_flat, X_test_flat, y_train_flat, y_test_flat = train_test_split(X_flat, y, test_size=0.2, stratify=y ,random_state=42)
print("Train:", Counter(y_train_flat))
print("Test:", Counter(y_test_flat))


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

models = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "SVM": LinearSVC()
}

for name, model in models.items():
    print(f"\nTraining : {name}")
    start = time.time()
    model.fit(X_train_flat, y_train_flat)
    y_pred = model.predict(X_test_flat)
    duration = time.time() - start

    acc = accuracy_score(y_test_flat, y_pred)
    model_accuracies[name] = acc

    filename = f"{name.lower().replace(' ', '_')}_model.pkl"
    joblib.dump(model, filename)
    print(f"Saved {name} model to {filename}")


    print(f"\n--- {name} ---")
    print(f" Accuracy: {acc:.2f}")
    print("Classification Report:")
    print(classification_report(y_test_flat, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_flat, y_pred))
    print(f"\n {name} completed in {duration:.2f} seconds.")

    

'''y = np.array(y)
# Step 1: Encode class labels (strings → integers)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Step 2: Split encoded labels for CNN
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded)

# Step 3: One-hot encode the labels
y_cat = to_categorical(y_encoded)
y_train_cat = to_categorical(y_train_cnn)
y_test_cat = to_categorical(y_test_cnn)

# === STEP 4: Train CNN ===
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128,128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y_cat.shape[1], activation='softmax')
])

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.fit(X_train_cnn, y_train_cat, epochs=10, batch_size=32, validation_data=(X_test_cnn, y_test_cat))

loss, acc = cnn.evaluate(X_test_cnn, y_test_cat)
print(f"\n--- CNN ---\nTest Accuracy: {acc:.2f}")



# Predict probabilities
y_pred_probs = cnn.predict(X_test_cnn)

# Convert probabilities to class indices
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test_cat, axis=1)


print("Classification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=encoder.classes_))


cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - CNN")
plt.show()


'# Count class labels
class_counts = Counter(y)  # y = original label list (before encoding)

# Bar plot
plt.figure(figsize=(7, 5))
plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
plt.title("📊 Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.grid(axis='y')
plt.show()

import random

plt.figure(figsize=(12, 4))
for idx, class_name in enumerate(set(y)):
    # Pick a random image from each class
    class_index = np.where(np.array(y) == class_name)[0]
    sample_img = X[random.choice(class_index)]
    
    plt.subplot(1, 3, idx + 1)
    plt.hist(sample_img.ravel(), bins=50, color='purple')
    plt.title(f"Pixel Intensity - {class_name}")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

class_counts = Counter(y)
import numpy as np
import pandas as pd

# Flatten all images: (N, 64, 64, 3) → (N, 12288)
X_flat = X.reshape(len(X), -1)

# Convert to DataFrame for statistics
df_pixels = pd.DataFrame(X_flat)

# Descriptive statistics for all pixels
stats_summary = df_pixels.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
stats_summary.index.name = "Pixel Index"
print(" Descriptive Statistics (All Pixels):")
print(stats_summary.round(2))


df_stats = pd.DataFrame.from_dict(class_counts, orient='index', columns=['Count'])
df_stats['Percentage'] = (df_stats['Count'] / df_stats['Count'].sum()) * 100
print("Descriptive Statistics:\n")
print(df_stats.round(2))



plt.tight_layout()
plt.show()


from sklearn.decomposition import PCA
import seaborn as sns

X_flat = X.reshape(len(X), -1)
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_flat)

df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(10)])
df_pca["Label"] = y_encoded

# Correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df_pca.corr(), annot=True, cmap="coolwarm")
plt.title(" PCA Component Correlation")
plt.show()'''



