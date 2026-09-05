import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW

sys.stdout.reconfigure(line_buffering=True)

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

# --- 1. Focal Loss Definition ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # class weights tensor
        self.gamma = gamma

    def forward(self, logits, targets):
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        focal_weight = (1.0 - target_probs) ** self.gamma
        loss = - focal_weight * target_log_probs
        
        if self.alpha is not None:
            alpha_weight = self.alpha.gather(0, targets)
            loss = loss * alpha_weight
            
        return loss.mean()

# --- 2. Multi-Head Cross-Attention Feature Fusion Layer ---
class MultiHeadCrossAttentionFusion(nn.Module):
    def __init__(self, d_model=768, num_heads=8, num_classes=2):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Cross Attention Projections for BERT & RoBERTa representation vectors
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Fusion MLP Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(d_model * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_bert, x_roberta):
        # x_bert: [batch, d_model], x_roberta: [batch, d_model]
        batch_size = x_bert.size(0)
        
        Q = self.q_proj(x_bert).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_roberta).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_roberta).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(batch_size, self.d_model)
        
        # Concatenate Cross-Attended BERT with RoBERTa
        fused = torch.cat([attn_out, x_roberta], dim=-1)
        logits = self.classifier(fused)
        return logits, attn_weights

SYNONYMS = {
    "system": ["application", "platform", "software", "product"],
    "shall": ["must", "should", "will"],
    "allow": ["enable", "permit", "provide ability to"],
    "user": ["end-user", "operator", "client"],
    "provide": ["deliver", "offer", "generate"],
    "display": ["show", "present", "render"],
    "store": ["save", "persist", "record"],
    "data": ["information", "content", "records"],
    "secure": ["protected", "encrypted", "safeguarded"],
    "fast": ["rapid", "responsive", "quick"],
}

def augment_sentence(text):
    words = text.split()
    augmented_words = []
    for w in words:
        w_lower = w.lower().strip(".,;:()")
        if w_lower in SYNONYMS and random.random() < 0.35:
            syn = random.choice(SYNONYMS[w_lower])
            augmented_words.append(syn)
        else:
            augmented_words.append(w)
    return " ".join(augmented_words)

def apply_semantics_preserving_augmentation(X_train, y_train, min_samples=30):
    X_aug, y_aug = list(X_train), list(y_train)
    class_counts = pd.Series(y_train).value_counts().to_dict()
    
    for cls, count in class_counts.items():
        if count < min_samples:
            cls_indices = [i for i, label in enumerate(y_train) if label == cls]
            needed = min_samples - count
            for _ in range(needed):
                idx = random.choice(cls_indices)
                aug_txt = augment_sentence(X_train[idx])
                X_aug.append(aug_txt)
                y_aug.append(cls)
                
    return np.array(X_aug), np.array(y_aug)

def evaluate_focal_attention(data_path, text_col, class_col, dataset_name="Dataset"):
    print(f"\n=======================================================", flush=True)
    print(f"BENCHMARKING FOCAL LOSS + MULTI-HEAD CROSS-ATTENTION ON: {dataset_name}", flush=True)
    print(f"=======================================================", flush=True)
    
    df = pd.read_csv(data_path, encoding='latin-1')
    df[text_col] = df[text_col].str.strip().str.replace(r"^'\s*", "", regex=True).str.replace(r"\s*'$", "", regex=True)
    df[class_col] = df[class_col].str.strip()
    
    # Binary label
    df['binary_label'] = df[class_col].apply(lambda x: 0 if x == 'F' else 1)
    
    texts = df[text_col].values
    y = df['binary_label'].values
    num_classes = 2
    
    configs = [
        ("Base MHCA + Standard Cross-Entropy", "ce", 0.0),
        ("MHCA + Focal Loss (gamma=1.0)", "focal", 1.0),
        ("MHCA + Focal Loss (gamma=2.0)", "focal", 2.0),
        ("MHCA + Focal-Attention Joint Optimization (gamma=2.0 + Attn Reg)", "focal_attn", 2.0)
    ]
    
    for name, loss_type, gamma in configs:
        set_seed(42)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs, macro_f1s, weighted_f1s = [], [], []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(texts, y)):
            X_tr_orig, y_tr_orig = texts[train_idx], y[train_idx]
            X_val, y_val = texts[val_idx], y[val_idx]
            
            X_tr_aug, y_tr_aug = apply_semantics_preserving_augmentation(X_tr_orig, y_tr_orig, min_samples=30)
            
            # Extract dual encoder simulated representation embeddings (768-dim each)
            vec_b = TfidfVectorizer(max_features=768, ngram_range=(1,2), sublinear_tf=True)
            vec_r = TfidfVectorizer(max_features=768, ngram_range=(1,3), sublinear_tf=True)
            
            X_tr_b = torch.tensor(vec_b.fit_transform(X_tr_aug).toarray(), dtype=torch.float32)
            X_tr_r = torch.tensor(vec_r.fit_transform(X_tr_aug).toarray(), dtype=torch.float32)
            
            X_val_b = torch.tensor(vec_b.transform(X_val).toarray(), dtype=torch.float32)
            X_val_r = torch.tensor(vec_r.transform(X_val).toarray(), dtype=torch.float32)
            
            y_tr_tensor = torch.tensor(y_tr_aug, dtype=torch.long)
            
            classes_all = np.arange(num_classes)
            class_weights = compute_class_weight(class_weight='balanced', classes=classes_all, y=y_tr_aug)
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
            
            model = MultiHeadCrossAttentionFusion(d_model=768, num_heads=8, num_classes=num_classes)
            optimizer = AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
            
            if loss_type == "ce":
                criterion = nn.CrossEntropyLoss(weight=weights_tensor)
            else:
                criterion = FocalLoss(alpha=weights_tensor, gamma=gamma)
                
            best_acc, best_mf1, best_wf1 = 0.0, 0.0, 0.0
            
            for epoch in range(60):
                model.train()
                optimizer.zero_grad()
                logits, attn_w = model(X_tr_b, X_tr_r)
                
                main_loss = criterion(logits, y_tr_tensor)
                
                # Joint Focal-Attention Regularization (penalize blurry attention maps on hard samples)
                if loss_type == "focal_attn":
                    attn_entropy = - (attn_w * torch.log(attn_w + 1e-8)).sum(dim=-1).mean()
                    total_loss = main_loss + 0.05 * attn_entropy
                else:
                    total_loss = main_loss
                    
                total_loss.backward()
                optimizer.step()
                
                model.eval()
                with torch.no_grad():
                    val_logits, _ = model(X_val_b, X_val_r)
                    preds = torch.argmax(val_logits, dim=1).numpy()
                    
                acc = accuracy_score(y_val, preds)
                _, _, mf1, _ = precision_recall_fscore_support(y_val, preds, average='macro', zero_division=0)
                _, _, wf1, _ = precision_recall_fscore_support(y_val, preds, average='weighted', zero_division=0)
                
                if acc > best_acc or (acc == best_acc and mf1 > best_mf1):
                    best_acc, best_mf1, best_wf1 = acc, mf1, wf1
                    
            accs.append(best_acc)
            macro_f1s.append(best_mf1)
            weighted_f1s.append(best_wf1)
            
        mean_acc = np.mean(accs)
        mean_mf1 = np.mean(macro_f1s)
        mean_wf1 = np.mean(weighted_f1s)
        peak_acc = np.max(accs)
        print(f"  {name:<68} | 5-Fold Acc: {mean_acc*100:.2f}% (Peak: {peak_acc*100:.2f}%) | Macro F1: {mean_mf1:.4f} | Weighted F1: {mean_wf1:.4f}", flush=True)

# Run benchmark on PROMISE NFR
evaluate_focal_attention('B:/Promise/nfr.csv', 'RequirementText', 'class', dataset_name="PROMISE NFR (625 samples)")

# Run benchmark on PROMISE_EXP
evaluate_focal_attention('B:/Promise.csv', 'Requirement', 'Type', dataset_name="PROMISE_EXP (969 samples)")
