# ============================================================
# Part 4: Data Visualization & Machine Learning
# Theme : Student Performance Analysis & Prediction
# Author: Gaurav Anand Shukla | ID: BITSoM_BA_25111017
# File  : part4_visualization_ml.py
# ============================================================
# Run this script after saving students.csv in the same folder.
# pip install pandas matplotlib seaborn scikit-learn
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (safe for scripts)
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# Save the dataset to students.csv
# ============================================================

csv_data = """name,math,science,english,history,pe,attendance_pct,study_hours_per_day,passed
Alice,88,92,76,80,95,92,4.5,1
Bob,42,55,48,50,60,65,1.2,0
Charlie,75,70,80,68,88,85,3.0,1
Diana,95,98,91,89,97,98,6.0,1
Eve,38,42,50,45,55,58,0.8,0
Frank,60,65,72,58,70,78,2.5,1
Grace,55,48,44,52,62,60,1.5,0
Henry,82,79,85,77,90,88,4.0,1
Iris,70,74,68,65,78,80,3.5,1
Jack,30,35,40,28,45,50,0.5,0
Karen,65,60,70,62,75,72,2.8,1
Liam,48,52,44,55,58,62,1.8,0
Mia,91,94,88,92,96,95,5.5,1
Noah,58,62,55,60,68,70,2.0,0
Olivia,78,75,82,70,85,84,3.8,1
"""

with open("students.csv", "w", encoding="utf-8") as f:
    f.write(csv_data.strip())
print("students.csv saved.")

# ============================================================
# TASK 1 — Data Exploration with Pandas (5 marks)
# ============================================================

print("\n" + "=" * 55)
print(" TASK 1 — Data Exploration with Pandas")
print("=" * 55)

df = pd.read_csv("students.csv")

# 1. First 5 rows
print("\n1. First 5 rows (.head()):")
print(df.head())

# 2. Shape and dtypes
print(f"\n2. Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\n   Data types (.dtypes):")
print(df.dtypes)

# 3. Summary statistics
print("\n3. Summary statistics (.describe()):")
print(df.describe())

# 4. Pass/Fail count
print("\n4. Pass/Fail count:")
print(df['passed'].value_counts().rename({1: 'Pass', 0: 'Fail'}))

# 5. Average score per subject for passing vs failing students
subject_cols = ['math', 'science', 'english', 'history', 'pe']
df['avg_score'] = df[subject_cols].mean(axis=1)

print("\n5. Average score per subject (Pass vs Fail):")
pass_avg = df[df['passed'] == 1][subject_cols].mean()
fail_avg = df[df['passed'] == 0][subject_cols].mean()
for subj in subject_cols:
    print(f"   {subj:<10} Pass avg: {pass_avg[subj]:.2f}   Fail avg: {fail_avg[subj]:.2f}")

# 6. Student with highest overall average
top_student = df.loc[df['avg_score'].idxmax()]
print(f"\n6. Student with highest overall average: {top_student['name']} ({top_student['avg_score']:.2f})")

# ============================================================
# TASK 2 — Data Visualization with Matplotlib (8 marks)
# ============================================================

print("\n" + "=" * 55)
print(" TASK 2 — Data Visualization with Matplotlib")
print("=" * 55)

# Plot 1 — Bar Chart: average score per subject
avg_per_subject = df[subject_cols].mean()
plt.figure(figsize=(8, 5))
plt.bar(avg_per_subject.index, avg_per_subject.values, color='steelblue', edgecolor='black')
plt.title("Average Score per Subject (All Students)")
plt.xlabel("Subject")
plt.ylabel("Average Score")
plt.tight_layout()
plt.savefig("plot1_bar.png")
plt.show()
print("  plot1_bar.png saved.")

# Plot 2 — Histogram: distribution of math scores (5 bins) + mean line
plt.figure(figsize=(7, 5))
plt.hist(df['math'], bins=5, color='orange', edgecolor='black')
mean_math = df['math'].mean()
plt.axvline(mean_math, color='red', linestyle='--', label=f'Mean = {mean_math:.1f}')
plt.title("Distribution of Math Scores")
plt.xlabel("Math Score")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("plot2_histogram.png")
plt.show()
print("  plot2_histogram.png saved.")

# Plot 3 — Scatter Plot: study_hours_per_day vs avg_score, coloured by passed
plt.figure(figsize=(7, 5))
pass_df = df[df['passed'] == 1]
fail_df = df[df['passed'] == 0]
plt.scatter(pass_df['study_hours_per_day'], pass_df['avg_score'],
            color='green', label='Pass', alpha=0.8)
plt.scatter(fail_df['study_hours_per_day'], fail_df['avg_score'],
            color='red', label='Fail', alpha=0.8)
plt.title("Study Hours per Day vs Average Score")
plt.xlabel("Study Hours per Day")
plt.ylabel("Average Score")
plt.legend()
plt.tight_layout()
plt.savefig("plot3_scatter.png")
plt.show()
print("  plot3_scatter.png saved.")

# Plot 4 — Box Plot: attendance_pct for Pass vs Fail students side by side
pass_attendance = df[df['passed'] == 1]['attendance_pct'].tolist()
fail_attendance = df[df['passed'] == 0]['attendance_pct'].tolist()
plt.figure(figsize=(6, 5))
plt.boxplot([pass_attendance, fail_attendance], labels=['Pass', 'Fail'])
plt.title("Attendance % Distribution: Pass vs Fail")
plt.xlabel("Result")
plt.ylabel("Attendance %")
plt.tight_layout()
plt.savefig("plot4_boxplot.png")
plt.show()
print("  plot4_boxplot.png saved.")

# Plot 5 — Line Plot: math and science scores for every student
plt.figure(figsize=(10, 5))
plt.plot(df['name'], df['math'],    marker='o', linestyle='-',  label='Math')
plt.plot(df['name'], df['science'], marker='s', linestyle='--', label='Science')
plt.title("Math vs Science Scores for Each Student")
plt.xlabel("Student")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("plot5_line.png")
plt.show()
print("  plot5_line.png saved.")

# ============================================================
# TASK 3 — Data Visualization with Seaborn (4 marks)
# ============================================================

print("\n" + "=" * 55)
print(" TASK 3 — Data Visualization with Seaborn")
print("=" * 55)

# Plot 6 — Seaborn bar plots: avg math and avg science score, split by passed
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(data=df, x='passed', y='math',    ax=axes[0])
axes[0].set_title("Average Math Score by Pass/Fail")
axes[0].set_xlabel("Passed (1=Yes, 0=No)")
axes[0].set_ylabel("Math Score")

sns.barplot(data=df, x='passed', y='science', ax=axes[1])
axes[1].set_title("Average Science Score by Pass/Fail")
axes[1].set_xlabel("Passed (1=Yes, 0=No)")
axes[1].set_ylabel("Science Score")

plt.tight_layout()
plt.savefig("plot6_seaborn_bar.png")
plt.show()
print("  plot6_seaborn_bar.png saved.")

# Plot 7 — Seaborn scatter + regression: attendance_pct vs avg_score by passed
plt.figure(figsize=(8, 5))
sns.regplot(data=df[df['passed'] == 1], x='attendance_pct', y='avg_score',
            label='Pass', color='green', scatter_kws={'alpha': 0.8})
sns.regplot(data=df[df['passed'] == 0], x='attendance_pct', y='avg_score',
            label='Fail', color='red', scatter_kws={'alpha': 0.8})
plt.title("Attendance % vs Average Score (with Regression Lines)")
plt.xlabel("Attendance %")
plt.ylabel("Average Score")
plt.legend()
plt.tight_layout()
plt.savefig("plot7_seaborn_scatter.png")
plt.show()
print("  plot7_seaborn_scatter.png saved.")

# Seaborn vs Matplotlib comparison comment:
# Seaborn was significantly easier for statistical plots — sns.barplot() and
# sns.regplot() automatically compute means, confidence intervals, and
# regression lines with very little code. Matplotlib required more manual
# work (grouping, manual bar heights, separate scatter calls). However,
# Matplotlib gives more low-level control over every plot element, which
# was useful for the custom line plot and histogram in Task 2.

# ============================================================
# TASK 4 — Machine Learning with scikit-learn (8 marks)
# ============================================================

print("\n" + "=" * 55)
print(" TASK 4 — Machine Learning with scikit-learn")
print("=" * 55)

feature_cols = ['math', 'science', 'english', 'history', 'pe', 'attendance_pct', 'study_hours_per_day']

# Step 1 — Prepare Data
X = df[feature_cols]
y = df['passed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features — fit on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Step 2 — Train a Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

train_acc = model.score(X_train_scaled, y_train)
print(f"\nTraining accuracy: {train_acc * 100:.2f}%")

# Step 3 — Evaluate the Model
y_pred   = model.predict(X_test_scaled)
test_acc = model.score(X_test_scaled, y_test)
print(f"Test accuracy    : {test_acc * 100:.2f}%")

print("\nPer-student predictions on the test set:")
for idx, (actual, predicted) in enumerate(zip(y_test, y_pred)):
    student_name_val = df.loc[X_test.index[idx], 'name']
    correct   = "✅ Correct" if actual == predicted else "❌ Wrong"
    act_label = "Pass" if actual == 1 else "Fail"
    pre_label = "Pass" if predicted == 1 else "Fail"
    print(f"  {student_name_val:<12} Actual: {act_label:<5}  Predicted: {pre_label:<5}  {correct}")

# Step 4 — Feature Importance
coefficients = model.coef_[0]
feature_importance = list(zip(feature_cols, coefficients))
feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

print("\nFeature importance (sorted by absolute coefficient):")
for feat, coef in feature_importance:
    direction = "→ Pass" if coef > 0 else "→ Fail"
    print(f"  {feat:<25} {coef:+.4f}  {direction}")

# Horizontal bar chart for feature importance
fig, ax = plt.subplots(figsize=(8, 5))
feats  = [f[0] for f in feature_importance]
coefs  = [f[1] for f in feature_importance]
colors = ['green' if c > 0 else 'red' for c in coefs]
ax.barh(feats, coefs, color=colors, edgecolor='black')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title("Logistic Regression Feature Coefficients")
ax.set_xlabel("Coefficient Value (green = Push to Pass, red = Push to Fail)")
ax.set_ylabel("Feature")
plt.tight_layout()
plt.savefig("plot8_feature_importance.png")
plt.show()
print("  plot8_feature_importance.png saved.")

# Step 5 — Predict for a New Student (Bonus)
new_student = [[75, 70, 68, 65, 80, 82, 3.2]]  # order matches feature_cols
new_student_scaled = scaler.transform(new_student)
prediction   = model.predict(new_student_scaled)[0]
probabilities = model.predict_proba(new_student_scaled)[0]

label = "Pass" if prediction == 1 else "Fail"
print(f"\nBonus — Prediction for new student {new_student[0]}:")
print(f"  Prediction  : {label}")
print(f"  Probability : Fail={probabilities[0]:.4f}  Pass={probabilities[1]:.4f}")
