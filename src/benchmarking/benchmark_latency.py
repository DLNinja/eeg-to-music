import time
import numpy as np
import torch
import warnings
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.model.emotion_classifier import EEGResNet

warnings.filterwarnings("ignore")

# MindToMusic Parameters
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


def setup_mind_to_music_pipeline():
    scaler_source = StandardScaler()
    scaler_target = StandardScaler()
    
    X_source = np.random.randn(200, 310)
    y_source = np.random.randint(0, 4, 200)
    X_cal = np.random.randn(50, 310)
    y_cal = np.random.randint(0, 4, 50)
    
    X_source = scaler_source.fit_transform(X_source)
    X_cal = scaler_target.fit_transform(X_cal)
    
    X_source_sel, y_source_sel = select_instances(X_source, y_source, X_cal, y_cal, total_k=100)
    ridge = get_sstm_mapping_model(X_source_sel, y_source_sel, X_cal, y_cal)
    
    clf_sstm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    clf_sstm.fit(X_source_sel, y_source_sel)
    
    return scaler_target, ridge, clf_sstm

def setup_resnet():
    device = torch.device('cpu')
    model = EEGResNet(num_classes=4).to(device)
    
    model_path = "models/best_model_EEGResNet_Shallow_V2_stft_smooth_lr5e-05.pt"
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    except Exception as e:
        pass
    
    model.eval()
    return model, device

def run_benchmark():
    print("Calibrating MindToMusic Pipeline (using exact notebook logic)...")
    scaler, ridge, svc = setup_mind_to_music_pipeline()
    
    print("Loading PyTorch ResNet...")
    resnet, device = setup_resnet()
    
    num_iterations = 1000
    print(f"\nSimulating real-time inference on {num_iterations} windows (1-second chunks, 62x5 features)...")
    
    test_chunks = [np.random.randn(1, 1, 62, 5).astype(np.float32) for _ in range(num_iterations)]
    
    # ==========================================
    # 1. Benchmark PyTorch ResNet
    # ==========================================
    resnet_latencies = []
    with torch.no_grad():
        for i in range(10):
            t = torch.tensor(test_chunks[i]).to(device)
            _ = resnet(t)
            
    with torch.no_grad():
        for chunk in test_chunks:
            start_time = time.perf_counter()
            t = torch.tensor(chunk).to(device)
            out = resnet(t)
            probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()
            end_time = time.perf_counter()
            resnet_latencies.append((end_time - start_time) * 1000)
            
    resnet_latencies = np.array(resnet_latencies)
    
    # ==========================================
    # 2. Benchmark MindToMusic (SSTM + SVC)
    # ==========================================
    m2m_latencies = []
    for i in range(10):
        c_flat = test_chunks[i].reshape(1, -1)
        c_scaled = scaler.transform(c_flat)
        c_mapped = ridge.predict(c_scaled)
        _ = svc.predict_proba(c_mapped)
        
    for chunk in test_chunks:
        start_time = time.perf_counter()
        
        c_flat = chunk.reshape(1, -1)
        c_scaled = scaler.transform(c_flat)
        c_mapped = ridge.predict(c_scaled)
        probs = svc.predict_proba(c_mapped)
        
        end_time = time.perf_counter()
        m2m_latencies.append((end_time - start_time) * 1000)
        
    m2m_latencies = np.array(m2m_latencies)
    
    # ==========================================
    # Results
    # ==========================================
    result_text = "="*50 + "\n"
    result_text += "BENCHMARK RESULTS (Time per 1-second chunk)\n"
    result_text += "="*50 + "\n"
    
    result_text += "\n--- EEGResNet (Shallow V2) ---\n"
    result_text += f"Average Latency: {np.mean(resnet_latencies):.2f} ms\n"
    result_text += f"Latency Std Dev (Jitter): {np.std(resnet_latencies):.2f} ms\n"
    result_text += f"99th Percentile: {np.percentile(resnet_latencies, 99):.2f} ms\n"
    
    result_text += "\n--- exact MindToMusic Pipeline (SSTM + SVC) ---\n"
    result_text += f"Average Latency: {np.mean(m2m_latencies):.2f} ms\n"
    result_text += f"Latency Std Dev (Jitter): {np.std(m2m_latencies):.2f} ms\n"
    result_text += f"99th Percentile: {np.percentile(m2m_latencies, 99):.2f} ms\n"
    
    print(result_text)
    
    with open("benchmark_results.txt", "w") as f:
        f.write(result_text)
    print("Results saved to benchmark_results.txt")

if __name__ == "__main__":
    run_benchmark()
