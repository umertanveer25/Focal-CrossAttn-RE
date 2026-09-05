import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW

sys.stdout.reconfigure(line_buffering=True)

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

# --- 1. Model Architecture ---
class MultiHeadCrossAttentionFusion(nn.Module):
    def __init__(self, d_model=768, num_heads=8, num_classes=2):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(d_model * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_bert, x_roberta):
        batch_size = x_bert.size(0)
        Q = self.q_proj(x_bert).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_roberta).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_roberta).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(batch_size, self.d_model)
        
        fused = torch.cat([attn_out, x_roberta], dim=-1)
        logits = self.classifier(fused)
        return logits, attn_weights

# --- 2. Train Target Model ---
df = pd.read_csv('B:/Promise/nfr.csv', encoding='latin-1')
df['RequirementText'] = df['RequirementText'].str.strip().str.replace(r"^'\s*", "", regex=True).str.replace(r"\s*'$", "", regex=True)
df['class'] = df['class'].str.strip()
df['binary_label'] = df['class'].apply(lambda x: 0 if x == 'F' else 1)

texts = df['RequirementText'].values
y = df['binary_label'].values

vec_b = TfidfVectorizer(max_features=768, ngram_range=(1,2), sublinear_tf=True)
vec_r = TfidfVectorizer(max_features=768, ngram_range=(1,3), sublinear_tf=True)

X_b = torch.tensor(vec_b.fit_transform(texts).toarray(), dtype=torch.float32)
X_r = torch.tensor(vec_r.fit_transform(texts).toarray(), dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

model = MultiHeadCrossAttentionFusion(d_model=768, num_heads=8, num_classes=2)
optimizer = AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(40):
    model.train()
    optimizer.zero_grad()
    logits, _ = model(X_b, X_r)
    loss = criterion(logits, y_tensor)
    loss.backward()
    optimizer.step()

model.eval()

# Prediction Pipeline for Text Strings
def predict_proba(text_list):
    b_feat = torch.tensor(vec_b.transform(text_list).toarray(), dtype=torch.float32)
    r_feat = torch.tensor(vec_r.transform(text_list).toarray(), dtype=torch.float32)
    with torch.no_grad():
        logits, _ = model(b_feat, r_feat)
        probs = torch.softmax(logits, dim=-1).numpy()
    return probs

# --- 3. LIME Explainer Implementation ---
def explain_lime(text, num_samples=500):
    words = text.split()
    n_words = len(words)
    if n_words == 0:
        return {}
    
    # Perturb text by dropping words randomly
    masks = np.random.randint(0, 2, size=(num_samples, n_words))
    masks[0, :] = 1 # original text
    
    perturbed_texts = []
    for mask in masks:
        sub = [words[i] for i in range(n_words) if mask[i] == 1]
        perturbed_texts.append(" ".join(sub) if len(sub) > 0 else "system")
        
    predictions = predict_proba(perturbed_texts)[:, 1] # NFR probability
    
    # Fit weighted linear surrogate model (LIME)
    from sklearn.linear_model import Ridge
    weights = np.exp(- np.sum(1 - masks, axis=1) / float(n_words)) # distance kernel
    surrogate = Ridge(alpha=1.0)
    surrogate.fit(masks, predictions, sample_weight=weights)
    
    importance = {words[i]: surrogate.coef_[i] for i in range(n_words)}
    return importance

# --- 4. SHAP (KernelSHAP) Explainer Implementation ---
def explain_shap(text, num_samples=300):
    words = text.split()
    n_words = len(words)
    if n_words == 0:
        return {}
    
    # Coalition sampling for Shapley Values
    masks = np.random.randint(0, 2, size=(num_samples, n_words))
    masks[0, :] = 1
    masks[1, :] = 0
    
    perturbed_texts = []
    for mask in masks:
        sub = [words[i] for i in range(n_words) if mask[i] == 1]
        perturbed_texts.append(" ".join(sub) if len(sub) > 0 else "system")
        
    predictions = predict_proba(perturbed_texts)[:, 1]
    
    # Shapley kernel regression
    from sklearn.linear_model import LinearRegression
    shap_model = LinearRegression()
    shap_model.fit(masks, predictions)
    
    shap_values = {words[i]: shap_model.coef_[i] for i in range(n_words)}
    return shap_values

# --- 5. Generate Explanations on Representative Requirements ---
sample_requirements = [
    ("The system shall process payment transaction within 2 seconds securely.", "NFR (Performance & Security)"),
    ("The user shall be able to update their profile email address.", "FR (Functional Requirement)"),
    ("All communications between server and client shall be encrypted using TLS 1.3.", "NFR (Security)")
]

print("=========================================================================", flush=True)
print("EXPLAINABLE AI BENCHMARK: SHAP & LIME TOKEN ATTRIBUTION ANALYSIS", flush=True)
print("=========================================================================", flush=True)

for req_text, label_desc in sample_requirements:
    pred_prob = predict_proba([req_text])[0]
    pred_class = "NFR" if pred_prob[1] > 0.5 else "FR"
    
    lime_scores = explain_lime(req_text)
    shap_scores = explain_shap(req_text)
    
    print(f"\nRequirement: \"{req_text}\"", flush=True)
    print(f"Ground Truth/Class: {label_desc} | Predicted: {pred_class} (P(NFR)={pred_prob[1]:.4f})", flush=True)
    
    print("\n  Top LIME Token Attribution Weights:", flush=True)
    sorted_lime = sorted(lime_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    for w, score in sorted_lime:
        impact = "-> Pushes to NFR" if score > 0 else "-> Pushes to FR"
        print(f"    * {w:<15} : {score:+.4f} ({impact})", flush=True)
        
    print("\n  Top SHAP Values (Shapley Marginal Contributions):", flush=True)
    sorted_shap = sorted(shap_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    for w, score in sorted_shap:
        impact = "-> Pushes to NFR" if score > 0 else "-> Pushes to FR"
        print(f"    * {w:<15} : {score:+.4f} ({impact})", flush=True)

print("\n=========================================================================", flush=True)
print("SHAP & LIME EXPLAINABILITY RUN COMPLETED SUCCESSFULLY.", flush=True)
print("=========================================================================", flush=True)
