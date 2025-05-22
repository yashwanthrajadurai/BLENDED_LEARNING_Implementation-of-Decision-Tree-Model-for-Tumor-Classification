# Implementation of Decision Tree Model for Tumor Classification
<H3>NAME: YASHWANTH RAJA DURAI V</H3>
<H3>REGISTER NO.: 212222040184</H3>
<H3>EX. NO.8</H3>

## AIM:
To implement and evaluate a Decision Tree model to classify tumors as benign or malignant using a dataset of lab test results.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Load Data**  
   Import the dataset to initiate the analysis.

2. **Explore Data**  
   Examine the dataset to identify patterns, distributions, and relationships.

3. **Select Features**  
   Determine the most important features to enhance model accuracy and efficiency.

4. **Split Data**  
   Separate the dataset into training and testing sets for effective validation.

5. **Train Model**  
   Use the training data to build and train the model.

6. **Evaluate Model**  
   Measure the model’s performance on the test data with relevant metrics.

## Program:
```py
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset from the provided URL
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/tumor.csv"
data = pd.read_csv(url)

# Step 2: Explore the dataset
# Display first few rows to understand the structure
print(data.head())

# Step 3: Select features and target variable
X = data.drop(columns=['Class'])  # Features: remove 'Class' column
y = data['Class']  # Target: 'Class' column for benign/malignant classification

# Step 4: Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)

# Print accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Visualize the Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

```

## Output:
![image](https://github.com/user-attachments/assets/27bb532d-bfec-4a25-8a69-2de1b7730d4e)
![image](https://github.com/user-attachments/assets/fdae2c18-b1b6-470d-a10d-6d6f2963e56a)


## Result:
Thus, the Decision Tree model was successfully implemented to classify tumors as benign or malignant, and the model’s performance was evaluated.
