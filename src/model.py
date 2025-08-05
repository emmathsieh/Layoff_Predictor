import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
import shap
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
nltk.download('punkt')

# 1. URLs for Google Sheets (export as CSV)
LAYOFFS_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1KpRyt3uMS7tELsPcEs02O8ipJcrEQlZSHeE81ZphjMo/"
    "export?format=csv&gid=1832121918"
)
GENAI_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1z7U0zM_-PXOhnbyqsSAtvQj6b3-gZVmjLxPiQ0jQ7IQ/"
    "export?format=csv&gid=1764913133"
)

# 2. Load data (unchanged)
df_layoffs = pd.read_csv(LAYOFFS_URL)
df_genai = pd.read_csv(GENAI_URL)

# 3. Standardize company names (unchanged)
for df in (df_layoffs, df_genai):
    df['Company'] = df['Company'].str.strip().str.upper()

# Add sentiment analysis function
def analyze_sentiment(text):
    if pd.isna(text) or str(text).strip() == '':
        return 0  # Neutral for missing/empty reviews
    analysis = TextBlob(str(text))
    return analysis.sentiment.polarity  # Range from -1 (negative) to 1 (positive)

# Apply sentiment analysis to the Employee Reviews column
df_genai['sentiment_score'] = df_genai['Employee Reviews'].apply(analyze_sentiment)

# 4. Parse dates (unchanged)
df_layoffs['Layoff_Date'] = pd.to_datetime(df_layoffs['Date'])
df_genai['Adoption_Year'] = pd.to_datetime(df_genai['Year Adopted'], format='%Y')

# 5. Create quarter period columns (unchanged)
df_layoffs['Quarter'] = df_layoffs['Layoff_Date'].dt.to_period('Q')
df_genai['Quarter'] = df_genai['Adoption_Year'].dt.to_period('Q')

# 6. Group by Company and Quarter, count layoffs per quarter (unchanged)
layoff_counts = df_layoffs.groupby(['Company', 'Quarter']).size().unstack(fill_value=0)
layoff_counts = layoff_counts.sort_index(axis=1)
layoff_history = layoff_counts.rolling(window=4, axis=1).sum()
layoff_history = layoff_history.stack().rename('layoff_history').reset_index()

# 7. Build GenAI adoption flag - modified to include sentiment
def cumulative_adoption_flag(df):
    # For each company, mark quarters at or after first adoption as 1, else 0
    # Also keep the sentiment score
    def flag_func(group):
        min_quarter = group['Quarter'].min()
        group['genai_flag'] = (group['Quarter'] >= min_quarter).astype(int)
        # For quarters before adoption, set sentiment to 0 (neutral)
        group.loc[group['Quarter'] < min_quarter, 'sentiment_score'] = 0
        return group
        
    return df.groupby('Company').apply(flag_func).reset_index(drop=True)

df_genai_sorted = df_genai.sort_values(['Company', 'Quarter'])
df_genai_flagged = cumulative_adoption_flag(df_genai_sorted)[['Company', 'Quarter', 'genai_flag', 'sentiment_score']]

# 8. Merge features and create target (unchanged except adding sentiment_score)
df_merged = (
    layoff_history
    .merge(df_genai_flagged, on=['Company', 'Quarter'], how='left')
    .fillna({'genai_flag': 0, 'sentiment_score': 0})  # Fill NA sentiment with neutral (0)
)

# Create target (unchanged)
df_merged['layoff_next_qtr'] = (
    df_merged.groupby('Company')['layoff_history']
    .shift(-1)
    .fillna(0)
    .gt(0)
    .astype(int)
)

# 9. Extract quarter number (unchanged)
df_merged['quarter_num'] = df_merged['Quarter'].dt.quarter.astype(int)
df_merged['quarter_sin'] = np.sin(2 * np.pi * df_merged['quarter_num'] / 4)
df_merged['quarter_cos'] = np.cos(2 * np.pi * df_merged['quarter_num'] / 4)

# 10. Prepare features X and target y - now including sentiment_score
feature_cols = ['genai_flag', 'layoff_history', 'quarter_sin', 'quarter_cos', 'sentiment_score']
X = df_merged[feature_cols].values
y = df_merged['layoff_next_qtr'].values

# 11. Train/validation split with stratification
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y
)

# 12. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# 13. Compute balanced class weights
class_weights_values = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)
class_weight_dict = dict(enumerate(class_weights_values))

# 14. Model definition with regularization and dropout
def make_model(input_dim):
    return models.Sequential([
        layers.InputLayer(input_shape=(input_dim,)),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

model = make_model(X_train.shape[1])

# 15. Adam optimizer and metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.Precision(name='precision')
    ]
)

# 16. Callbacks for early stopping and learning rate reduction
cb_early = callbacks.EarlyStopping(
    monitor='val_auc', mode='max', patience=15, restore_best_weights=True
)
cb_plateau = callbacks.ReduceLROnPlateau(
    monitor='val_auc', factor=0.5, patience=5, verbose=1
)

# 17. Train model with class weights
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200, # ?
    batch_size=64,
    class_weight=class_weight_dict,
    callbacks=[cb_early, cb_plateau],
    verbose=1
)

# 18. Evaluate model on validation set
eval_results = model.evaluate(X_val, y_val, batch_size=64, verbose=0)
print(f"\nValidation Loss: {eval_results[0]:.4f}, AUC: {eval_results[1]:.4f}")

# 19. Predict probabilities and apply tuned threshold
y_pred_prob = model.predict(X_val)
optimal_threshold = 0.4  # let's just put this for now
y_pred_binary = (y_pred_prob > optimal_threshold).astype(int)

# 20. Classification report
print("\nClassification Report:")
print(classification_report(y_val, y_pred_binary, target_names=['No Layoff', 'Layoff'], zero_division=0))

# 21. SHAP Interpretation with DeepExplainer
# Use a subset of training data as background for SHAP
background = X_train[np.random.choice(X_train.shape[0], min(100, X_train.shape[0]), replace=False)]
explainer = shap.DeepExplainer(model, background)

# Compute SHAP values on a subset of validation set for speed
shap_values = explainer.shap_values(X_val[:100])

# shap_values is a list with one array for binary classification - take the first
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# Plot SHAP summary
# shap.summary_plot(
#     shap_values,
#     X_val[:100],
#     feature_names=feature_cols,
#     plot_type='dot',
#     show=True
# )

# 22. Plot training history: AUC and Loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['auc'], label='Train AUC')
plt.plot(history.history['val_auc'], label='Val AUC')
plt.title('Model AUC')
plt.ylabel('AUC')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()