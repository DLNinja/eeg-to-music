
import os
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.model.emotion_classifier import EEGResNet

warnings.filterwarnings("ignore")

# MindToMusic parameters
K_INSTANCES = 10000
RHO = 0.1

def select_instances(X_source, y_source, X_target_cal, y_target_cal, total_k=K_INSTANCES):
    if len(np.unique(y_target_cal)) < 2:
        indices = np.random.choice(len(X_source), min(len(X_source), total_k), replace=False)
        return X_source[indices], y_source[indices]

    c0 = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
    c0.fit(X_target_cal, y_target_cal)
    class_map = {label: idx for idx, label in enumerate(c0.classes_)}

    probas = c0.predict_proba(X_source)
    selected_indices = []
    unique_classes = np.unique(y_source)
    n_classes = len(unique_classes)
    k_per_class = max(1, total_k // n_classes)

    for c in unique_classes:
        true_class_indices = np.where(y_source == c)[0]
        if len(true_class_indices) == 0: continue
        if c in class_map:
            col_idx = class_map[c]
            class_scores = probas[true_class_indices, col_idx]
        else:
            class_scores = np.zeros(len(true_class_indices))

        sorted_local_indices = np.argsort(class_scores)[::-1]
        actual_k = min(len(sorted_local_indices), k_per_class)
        top_k_local = sorted_local_indices[:actual_k]
        selected_indices.extend(true_class_indices[top_k_local])

    selected_indices = np.array(selected_indices)
    return X_source[selected_indices], y_source[selected_indices]

def get_sstm_mapping_model(X_source_sel, y_source_sel, X_target_cal, y_target_cal):
    classes = np.unique(y_target_cal)
    mu_source = {}
    mu_target = {}

    for c in classes:
        s_data = X_source_sel[y_source_sel == c]
        mu_source[c] = np.mean(s_data, axis=0) if len(s_data) > 0 else np.zeros(X_source_sel.shape[1])
        t_data = X_target_cal[y_target_cal == c]
        mu_target[c] = np.mean(t_data, axis=0) if len(t_data) > 0 else np.zeros(X_target_cal.shape[1])

    O_train = []
    S_train = []
    for i, t_sample in enumerate(X_target_cal):
        label = y_target_cal[i]
        o_i = mu_source[label] + RHO * (t_sample - mu_target[label])
        O_train.append(o_i)
        S_train.append(mu_source[label])

    O_train = np.array(O_train)
    S_train = np.array(S_train)

    ridge = Ridge(alpha=1.0)
    ridge.fit(O_train, S_train)
    return ridge


def build_class_windows(y_target, min_class_count):
    # For each emotion type, split sorted indices into non-overlapping
    # windows of size `min_class_count` (which is the Fear class count).
    # If remainder exists, a final smaller window is included.
    # Returns a dict:
    #  { class_label: [ [indices_window_0], [indices_window_1], ... ] }

    windows = {}
    for c in np.unique(y_target):
        idx = np.sort(np.where(y_target == c)[0])
        class_windows = []
        for start in range(0, len(idx), min_class_count):
            window = idx[start:start + min_class_count]
            class_windows.append(window)
        windows[c] = class_windows
    return windows


def run_single_iteration(iter_num, total_iters, class_indices,
                         X_target, X_target_flat, y_target,
                         X_source_scaled, y_source,
                         model_path, cal_ratio=0.40):
    # Run one iteration of the benchmark with the given class indices.
    # Returns (mind2music_acc, resnet_acc, y_test, mind2music_preds, resnet_preds).

    window_size = min(len(idx) for idx in class_indices.values())

    X_cal_flat_list, X_test_flat_list = [], []
    y_cal_list, y_test_list = [], []
    X_cal_3d_list, X_test_3d_list = [], []

    for c, idx in class_indices.items():
        if len(idx) > window_size:
            np.random.seed(42 + iter_num)
            idx = np.sort(np.random.choice(idx, window_size, replace=False))

        split_point = int(len(idx) * cal_ratio)

        cal_idx = idx[:split_point]
        test_idx = idx[split_point:]

        X_cal_flat_list.append(X_target_flat[cal_idx])
        X_test_flat_list.append(X_target_flat[test_idx])
        y_cal_list.append(y_target[cal_idx])
        y_test_list.append(y_target[test_idx])
        X_cal_3d_list.append(X_target[cal_idx])
        X_test_3d_list.append(X_target[test_idx])

    X_cal_flat = np.concatenate(X_cal_flat_list)
    X_test_flat = np.concatenate(X_test_flat_list)
    y_cal = np.concatenate(y_cal_list)
    y_test = np.concatenate(y_test_list)
    X_cal_3d = np.concatenate(X_cal_3d_list)
    X_test_3d = np.concatenate(X_test_3d_list)

    # MindToMusic SSTM-IS 
    scaler_target = StandardScaler()
    X_cal_scaled = scaler_target.fit_transform(X_cal_flat)
    X_test_scaled = scaler_target.transform(X_test_flat)

    X_source_sel, y_source_sel = select_instances(
        X_source_scaled, y_source, X_cal_scaled, y_cal, total_k=1000
    )

    ridge = get_sstm_mapping_model(X_source_sel, y_source_sel, X_cal_scaled, y_cal)
    X_test_mapped = ridge.predict(X_test_scaled)

    clf_sstm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    clf_sstm.fit(X_source_sel, y_source_sel)

    m2m_preds = clf_sstm.predict(X_test_mapped)
    m2m_acc = accuracy_score(y_test, m2m_preds)

    # ResNet Calibrated 
    device = torch.device('cpu')
    model = EEGResNet(num_classes=4).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(checkpoint["model_state"])
    except Exception:
        pass  

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    X_cal_flat_resnet = X_cal_3d.reshape(X_cal_3d.shape[0], -1)
    X_test_flat_resnet = X_test_3d.reshape(X_test_3d.shape[0], -1)

    scaler_resnet = StandardScaler()
    X_cal_scaled_resnet = scaler_resnet.fit_transform(X_cal_flat_resnet).reshape(X_cal_3d.shape)
    X_test_scaled_resnet = scaler_resnet.transform(X_test_flat_resnet).reshape(X_test_3d.shape)

    X_cal_pt = torch.tensor(X_cal_scaled_resnet).unsqueeze(1).float().to(device)
    y_cal_pt = torch.tensor(y_cal).long().to(device)
    X_test_pt = torch.tensor(X_test_scaled_resnet).unsqueeze(1).float().to(device)

    cal_dataset = TensorDataset(X_cal_pt, y_cal_pt)
    cal_loader = DataLoader(cal_dataset, batch_size=32, shuffle=True)

    for epoch in range(15):
        for batch_X, batch_y in cal_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_loader = DataLoader(TensorDataset(X_test_pt, torch.zeros(len(X_test_pt)).long()), batch_size=32, shuffle=False)
        resnet_preds = []
        for batch_X, _ in test_loader:
            resnet_preds.extend(torch.argmax(model(batch_X), dim=1).cpu().numpy())
    resnet_preds = np.array(resnet_preds)

    resnet_acc = accuracy_score(y_test, resnet_preds)

    print(f"  Iter {iter_num}/{total_iters} | "
          f"Window={window_size}/class | "
          f"SSTM: {m2m_acc*100:.1f}% | ResNet: {resnet_acc*100:.1f}%")

    return m2m_acc, resnet_acc, y_test, m2m_preds, resnet_preds


# ─── Main Test Pipeline ────────────────────────────────────────────────────

def test_pipeline():
    print("=" * 70)
    print("  SLIDING-WINDOW CROSS-VALIDATION BENCHMARK (Self-Recorded BDF)")
    print("=" * 70)

    # Load BDF data
    print("\nLoading Target Data (Extracted BDF features)...")
    bdf_data = np.load("models/bdf/extracted_features.npz")
    X_target = bdf_data['X']       # (N, 62, 5)
    y_target = bdf_data['y']       # (N,)
    X_target_flat = X_target.reshape(X_target.shape[0], -1)

    # Class distribution
    emotions = ['Neutral', 'Sad', 'Fear', 'Happy']
    class_counts = {c: int(np.sum(y_target == c)) for c in np.unique(y_target)}
    min_class = min(class_counts, key=class_counts.get)
    min_count = class_counts[min_class]

    print(f"\nClass distribution:")
    for c in sorted(class_counts):
        n = class_counts[c]
        n_windows = len(range(0, n, min_count))
        marker = " <- bottleneck (fixed)" if c == min_class else f" -> {n_windows} windows of {min_count}"
        print(f"  {emotions[int(c)]}: {n} samples{marker}")

    # Non-overlapping windows
    all_windows = build_class_windows(y_target, min_count)

    # All combinations of windows (one per class)
    class_labels = sorted(all_windows.keys())
    window_lists = [list(range(len(all_windows[c]))) for c in class_labels]
    all_combos = list(itertools.product(*window_lists))

    print(f"\nTotal iterations: {len(all_combos)} "
          f"({'x'.join(str(len(all_windows[c])) for c in class_labels)} combinations)")

    # Load source SEED data (shared across iterations)
    print("\nLoading Source SEED Data...")
    source_data = np.load("models/de_stft_smooth.npz")
    X_source = source_data['X']
    if X_source.ndim == 3:
        X_source = X_source.reshape(X_source.shape[0], -1)
    y_source = source_data['y']

    np.random.seed(42)
    indices = np.random.choice(len(X_source), 5000, replace=False)
    X_source = X_source[indices]
    y_source = y_source[indices]

    scaler_source = StandardScaler()
    X_source_scaled = scaler_source.fit_transform(X_source)

    model_path = "models/best_model_stft_smooth.pt"

    # Run all iterations
    print(f"\n{'-'*70}")
    print("Running iterations...")
    print(f"{'-'*70}")

    all_m2m_accs = []
    all_resnet_accs = []
    all_confusion_m2m = []
    all_confusion_resnet = []

    for combo_idx, combo in enumerate(all_combos):
        # Build the index dict for this combination
        class_indices = {}
        for i, c in enumerate(class_labels):
            class_indices[c] = all_windows[c][combo[i]]

        m2m_acc, resnet_acc, y_test, m2m_preds, resnet_preds = run_single_iteration(
            iter_num=combo_idx + 1,
            total_iters=len(all_combos),
            class_indices=class_indices,
            X_target=X_target,
            X_target_flat=X_target_flat,
            y_target=y_target,
            X_source_scaled=X_source_scaled,
            y_source=y_source,
            model_path=model_path
        )

        all_m2m_accs.append(m2m_acc)
        all_resnet_accs.append(resnet_acc)
        all_confusion_m2m.append(confusion_matrix(y_test, m2m_preds, labels=[0,1,2,3]))
        all_confusion_resnet.append(confusion_matrix(y_test, resnet_preds, labels=[0,1,2,3]))

    # ─── Aggregate Results ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  AGGREGATED RESULTS")
    print(f"{'='*70}")

    m2m_mean = np.mean(all_m2m_accs) * 100
    m2m_std  = np.std(all_m2m_accs) * 100
    res_mean = np.mean(all_resnet_accs) * 100
    res_std  = np.std(all_resnet_accs) * 100

    print(f"\n  MindToMusic SSTM:    {m2m_mean:.2f}% +/- {m2m_std:.2f}%  (n={len(all_m2m_accs)} iterations)")
    print(f"  Calibrated ResNet:   {res_mean:.2f}% +/- {res_std:.2f}%  (n={len(all_resnet_accs)} iterations)")
    print(f"\n  Per-iteration breakdown:")
    for i, (m, r) in enumerate(zip(all_m2m_accs, all_resnet_accs)):
        print(f"    Iter {i+1}: SSTM={m*100:.1f}%  ResNet={r*100:.1f}%")

    # ─── Averaged Confusion Matrices ───────────────────────────────────
    avg_cm_m2m = np.mean(all_confusion_m2m, axis=0)
    avg_cm_resnet = np.mean(all_confusion_resnet, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(avg_cm_m2m, annot=True, fmt='.1f', cmap='Blues', ax=axes[0],
                xticklabels=emotions, yticklabels=emotions)
    axes[0].set_title(f"MindToMusic SSTM (Avg)\n{m2m_mean:.1f}% ± {m2m_std:.1f}%")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True Emotion")

    sns.heatmap(avg_cm_resnet, annot=True, fmt='.1f', cmap='Reds', ax=axes[1],
                xticklabels=emotions, yticklabels=emotions)
    axes[1].set_title(f"Calibrated ResNet (Avg)\n{res_mean:.1f}% ± {res_std:.1f}%")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True Emotion")

    plt.suptitle(f"Sliding-Window Cross-Validation ({len(all_combos)} iterations)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("src/benchmarking/emotion_confusion_sliding_window.png", dpi=150)
    print(f"\nSaved confusion matrices to src/benchmarking/emotion_confusion_sliding_window.png")

    # ─── Accuracy Distribution Plot ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = [0, 1]
    bp = ax.boxplot([np.array(all_m2m_accs)*100, np.array(all_resnet_accs)*100],
                     positions=positions, widths=0.5, patch_artist=True)

    bp['boxes'][0].set_facecolor('#4A90D9')
    bp['boxes'][1].set_facecolor('#D94A4A')

    for i, (accs, color) in enumerate([(all_m2m_accs, '#2C5F9E'), (all_resnet_accs, '#9E2C2C')]):
        jitter = np.random.uniform(-0.1, 0.1, len(accs))
        ax.scatter([i + j for j in jitter], np.array(accs)*100, color=color, alpha=0.6, s=30, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(['MindToMusic SSTM', 'Calibrated ResNet'])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Accuracy Distribution Across {len(all_combos)} Sliding-Window Iterations')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("src/benchmarking/accuracy_distribution_sliding_window.png", dpi=150)
    print(f"Saved accuracy distribution to src/benchmarking/accuracy_distribution_sliding_window.png")

    # ─── Save raw results ──────────────────────────────────────────────
    results_path = "src/benchmarking/sliding_window_results.txt"
    with open(results_path, 'w') as f:
        f.write("Sliding-Window Cross-Validation Results\n")
        f.write(f"{'='*50}\n")
        f.write(f"Fear (bottleneck): {min_count} samples\n")
        f.write(f"Total iterations: {len(all_combos)}\n\n")
        f.write(f"MindToMusic SSTM:  {m2m_mean:.2f}% ± {m2m_std:.2f}%\n")
        f.write(f"Calibrated ResNet: {res_mean:.2f}% ± {res_std:.2f}%\n\n")
        for i, (m, r) in enumerate(zip(all_m2m_accs, all_resnet_accs)):
            f.write(f"Iter {i+1}: SSTM={m*100:.1f}%  ResNet={r*100:.1f}%\n")
    print(f"Saved raw results to {results_path}")


if __name__ == "__main__":
    test_pipeline()
