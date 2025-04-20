import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # for saving the model

# === 1. Load dataset ===
df = pd.read_csv("batch_results_with_best_algo.csv")

# === 2. Define features and target ===
X = df[['BestAlgo_ATAT', 'BestAlgo_AWT', 'BestAlgo_ART']]
y = df['BestAlgo_by_Score']

# === 3. Split data into training and testing sets ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.7, random_state=42
)

# === 4. Train the Random Forest model ===
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# === 5. Save the trained model ===
joblib.dump(model, "random_forest_model.pkl")
print("Model saved as 'random_forest_model.pkl'")

# === 6. Make predictions on test data ===
y_pred = model.predict(X_test)

# === 7. Evaluate the model ===
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# === 8. Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# === 9. Visualize the confusion matrix ===
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# === 10. Feature Importances ===
importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\nFeature Importances:")
print(importances)