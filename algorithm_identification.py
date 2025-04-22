import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Generate synthetic dataset with patterns from different algorithms
def generate_dataset(samples=1000):
    np.random.seed(42)
    data = []
    labels = []

    for _ in range(samples):
        choice = np.random.choice(['sorted', 'random', 'reversed', 'linear', 'quadratic'])
        
        if choice == 'sorted':
            arr = np.sort(np.random.randint(1, 100, size=10))
        elif choice == 'random':
            arr = np.random.randint(1, 100, size=10)
        elif choice == 'reversed':
            arr = np.sort(np.random.randint(1, 100, size=10))[::-1]
        elif choice == 'linear':
            x = np.linspace(1, 10, 10)
            arr = 2 * x + 3
        elif choice == 'quadratic':
            x = np.linspace(1, 10, 10)
            arr = x**2 + 2*x + 1
        
        data.append(arr)
        labels.append(choice)
    
    return np.array(data), np.array(labels)

# Generate dataset
X, y = generate_dataset()

# Convert to DataFrame for better visualization
df = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(X.shape[1])])
df['Label'] = y

# Display dataset structure
print(df.head())

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
