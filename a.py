import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 數據預處理
df = pd.read_csv("合併後的數據.csv")
print("數據集列名:", df.columns)  # 打印列名以檢查

X = df[['速度']]
y = df['成績']

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 標準化特徵
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. 模型選擇和訓練
models = {
    '線性回歸': LinearRegression(),
    'Ridge回歸': Ridge(),
    'Lasso回歸': Lasso()
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    results[name] = {'MSE': mse, 'R2': r2, 'CV_R2_mean': np.mean(cv_scores)}

# 3. 模型評估
for name, metrics in results.items():
    print(f"{name}:")
    print(f"  MSE: {metrics['MSE']:.4f}")
    print(f"  R2: {metrics['R2']:.4f}")
    print(f"  交叉驗證 R2: {metrics['CV_R2_mean']:.4f}")
    print()

# 選擇最佳模型
best_model_name = max(results, key=lambda x: results[x]['R2'])
best_model = models[best_model_name]

# 使用最佳模型進行最終預測
final_predictions = best_model.predict(X_test_scaled)

# 4. 繪製結果
plt.figure(figsize=(12, 8))
sns.scatterplot(x=X_test['速度'], y=y_test, color='blue', label='實際值')
sns.scatterplot(x=X_test['速度'], y=final_predictions, color='red', label='預測值')

# 添加回歸線
sns.regplot(x=X_test['速度'], y=final_predictions, scatter=False, color='green', label='回歸線')

plt.xlabel('速度')
plt.ylabel('成績')
plt.title(f'最佳模型: {best_model_name}')
plt.legend()
plt.show()

# 打印中文輸出
print(f"最佳模型: {best_model_name}")
print(f"MSE: {results[best_model_name]['MSE']:.4f}")
print(f"R2: {results[best_model_name]['R2']:.4f}")
print(f"交叉驗證 R2: {results[best_model_name]['CV_R2_mean']:.4f}")