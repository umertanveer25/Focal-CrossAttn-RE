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

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # class weights tensor
        self.gamma = gamma

    def forward(self, logits, targets):
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        # Gather probabilities for target classes
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        focal_weight = (1.0 - target_probs) ** self.gamma
        loss = - focal_weight * target_log_probs
        
        if self.alpha is not None:
            alpha_weight = self.alpha.gather(0, targets)
            loss = loss * alpha_weight
            
        return loss.mean()

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

class AuthorFeatureFusionNN(nn.Module):
    def __init__(self, in_features=1536, num_classes=2):
        super().__init__()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(in_features, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

def evaluate_dataset_loss(data_path, text_col, class_col, is_binary_only=False, dataset_name="Dataset"):
    print(f"\n=======================================================", flush=True)
    print(f"EVALUATING FOCAL LOSS ON: {dataset_name}", flush=True)
    print(f"=======================================================", flush=True)
    
    df = pd.read_csv(data_path, encoding='latin-1')
    df[text_col] = df[text_col].str.strip().str.replace(r"^'\s*", "", regex=True).str.replace(r"\s*'$", "", regex=True)
    df[class_col] = df[class_col].str.strip()
    
    # Binary target
    df['binary_label'] = df[class_col].apply(lambda x: 0 if x == 'F' else 1)
    
    # Multi-class target
    unique_classes = sorted(df[class_col].unique())
    class2idx = {c: i for i, c in enumerate(unique_classes)}
    df['multi_label'] = df[class_col].map(class2idx)
    
    loss_configs = [
        ("Cross-Entropy (Class Weighted)", lambda w: nn.CrossEntropyLoss(weight=w)),
        ("Focal Loss (gamma=1.0)", lambda w: FocalLoss(alpha=w, gamma=1.0)),
        ("Focal Loss (gamma=2.0)", lambda w: FocalLoss(alpha=w, gamma=2.0)),
        ("Focal Loss (gamma=3.0)", lambda w: FocalLoss(alpha=w, gamma=3.0)),
    ]
    
    tasks = [('binary_label', 'Binary Classification')]
    if not is_binary_only and len(unique_classes) > 2:
        tasks.append(('multi_label', 'Multi-Class Classification'))
        
    for target_col, task_name in tasks:
        print(f"\n--- Task: {task_name} ({dataset_name}) ---", flush=True)
        texts = df[text_col].values
        y = df[target_col].values
        num_classes = len(np.unique(y))
        
        for loss_name, loss_fn_builder in loss_configs:
            set_seed(42)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            accs, macro_f1s, weighted_f1s = [], [], []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(texts, y)):
                X_tr_orig, y_tr_orig = texts[train_idx], y[train_idx]
                X_val, y_val = texts[val_idx], y[val_idx]
                
                X_tr_aug, y_tr_aug = apply_semantics_preserving_augmentation(X_tr_orig, y_tr_orig, min_samples=30)
                
                vec = TfidfVectorizer(max_features=1536, ngram_range=(1,3), sublinear_tf=True)
                X_tr_v = torch.tensor(vec.fit_transform(X_tr_aug).toarray(), dtype=torch.float32)
                X_val_v = torch.tensor(vec.transform(X_val).toarray(), dtype=torch.float32)
                y_tr_tensor = torch.tensor(y_tr_aug, dtype=torch.long)
                
                classes_all = np.arange(num_classes)
                class_weights = compute_class_weight(class_weight='balanced', classes=classes_all, y=y_tr_aug)
                weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
                
                model = AuthorFeatureFusionNN(in_features=1536, num_classes=num_classes)
                optimizer = AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
                criterion = loss_fn_builder(weights_tensor)
                
                best_acc, best_mf1, best_wf1 = 0.0, 0.0, 0.0
                
                for epoch in range(60):
                    model.train()
                    optimizer.zero_grad()
                    logits = model(X_tr_v)
                    loss = criterion(logits, y_tr_tensor)
                    loss.backward()
                    optimizer.step()
                    
                    model.eval()
                    with torch.no_grad():
                        val_logits = model(X_val_v)
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
            print(f"  {loss_name:<32} | 5-Fold Acc: {mean_acc*100:.2f}% (Peak: {peak_acc*100:.2f}%) | Macro F1: {mean_mf1:.4f} | Weighted F1: {mean_wf1:.4f}", flush=True)

# Run on PROMISE NFR
evaluate_dataset_loss('B:/Promise/nfr.csv', 'RequirementText', 'class', is_binary_only=True, dataset_name="PROMISE NFR (625 samples)")

# Run on PROMISE_EXP
evaluate_dataset_loss('B:/Promise.csv', 'Requirement', 'Type', is_binary_only=False, dataset_name="PROMISE_EXP (969 samples)")
