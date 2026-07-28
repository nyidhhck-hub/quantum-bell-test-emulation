import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# ==========================================================
# --- CONFIGURATION & FIXED CALIBRATION PARAMETERS ---
# ==========================================================
# Calibration parameters based on sensor noise-floor measurements
BEST_PROM = 0.25
BEST_HEIGHT = 0.06
BEST_SHIFT = 3
BEST_WINDOW = 2
BEST_PD = 18

# Experimental setup angles & ROI coordinates
alpha_angles = [-45, 0, 45, 90]
beta_angles = [-22.5, 22.5, 67.5, 112.5]
roi_coords = [(338, 375, 67, 56), (726, 369, 79, 57)]  # [Alice, Bob]
labels = ["Alice", "Bob"]
colors = ["b", "r"]


# ==========================================================
# --- HELPER FUNCTIONS ---
# ==========================================================
def extract_roi_signals(cap, coords):
    """Extracts red-channel intensity signals from video ROIs."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    results = [[] for _ in range(len(coords))]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for i, (x, y, w, h) in enumerate(coords):
            roi = frame[y:y + h, x:x + w]
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Red color thresholds in HSV
            mask1 = cv2.inRange(roi_hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(roi_hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
            red_mask = mask1 | mask2

            results[i].append(np.sum(red_mask))

    cap.release()
    return [np.array(results[0]), np.array(results[1])]


def shift_signal(arr, shift):
    """Applies temporal shift to compensate for channel delay."""
    if len(arr) == 0 or shift == 0:
        return arr.copy()
    if shift > 0:
        return np.pad(arr, (shift, 0), mode='constant', constant_values=0)[:-shift]
    return np.pad(arr, (0, -shift), mode='constant', constant_values=0)[-shift:]


def calculate_coincidences(peaks_a, peaks_b, window=2):
    """Calculates coincidences within a specified time window."""
    if len(peaks_a) == 0 or len(peaks_b) == 0:
        return 0
    count = 0
    used_b = set()
    for pa in peaks_a:
        for pb in peaks_b:
            if pb not in used_b and abs(pa - pb) <= window:
                count += 1
                used_b.add(pb)
                break
    return count


def compute_CHSH_S(counts_map):
    """Calculates CHSH parameter S and its statistical error delta_S."""
    def get_N(a, b):
        a_mod, b_mod = a % 360, b % 360
        if a_mod == 135: a_mod = -45
        if b_mod == 157.5: b_mod = -22.5
        return counts_map.get(f"N({a_mod},{b_mod})", 0)

    def E_and_var(a, b):
        N_ab = get_N(a, b)
        N_ap_bp = get_N(a + 90, b + 90)
        N_a_bp = get_N(a, b + 90)
        N_ap_b = get_N(a + 90, b)
        den = N_ab + N_ap_bp + N_a_bp + N_ap_b
        if den == 0:
            return 0.0, 0.0
        E_val = (N_ab + N_ap_bp - N_a_bp - N_ap_b) / den
        var_E = (1.0 - E_val**2) / den
        return E_val, var_E

    E1, v1 = E_and_var(0, 22.5)
    E2, v2 = E_and_var(0, 67.5)
    E3, v3 = E_and_var(45, 22.5)
    E4, v4 = E_and_var(45, 67.5)

    s_candidates = [
        E1 - E2 + E3 + E4,
        E1 + E2 + E3 - E4,
        E1 + E2 - E3 + E4,
        -E1 + E2 + E3 + E4
    ]

    best_idx = int(np.argmax([abs(s) for s in s_candidates]))
    max_S = s_candidates[best_idx]
    abs_S = abs(max_S)
    delta_S = np.sqrt(v1 + v2 + v3 + v4)

    return abs_S, delta_S


# ==========================================================
# --- STEP 1: RAW DATA EXTRACTION ---
# ==========================================================
print("--- Step 1: Extracting Signals from Video Files ---")
all_raw_signals = {}

for alpha in alpha_angles:
    for beta in beta_angles:
        video_path = os.path.join(f"ALPHA {alpha}", f"{beta}.mp4")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Warning: Could not open {video_path}")
            all_raw_signals[(alpha, beta)] = [np.array([]), np.array([])]
        else:
            all_raw_signals[(alpha, beta)] = extract_roi_signals(cap, roi_coords)

# Global normalization factors
global_max_alice = max([np.max(sig[0]) if len(sig[0]) > 0 else 1.0 for sig in all_raw_signals.values()]) or 1.0
global_max_bob = max([np.max(sig[1]) if len(sig[1]) > 0 else 1.0 for sig in all_raw_signals.values()]) or 1.0


# ==========================================================
# --- STEP 2: PEAK DETECTION & GRID PLOTTING ---
# ==========================================================
print("--- Step 2: Processing Signals & Plotting ---")
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 12), sharex=False, sharey=True)
counts_dict = {}

for row_idx, alpha in enumerate(alpha_angles):
    for col_idx, beta in enumerate(beta_angles):
        ax = axes[row_idx, col_idx]

        alice_raw = all_raw_signals[(alpha, beta)][0]
        bob_raw = all_raw_signals[(alpha, beta)][1]

        norm_a = alice_raw / global_max_alice
        norm_b = shift_signal(bob_raw, BEST_SHIFT) / global_max_bob

        peaks_a, _ = find_peaks(norm_a, prominence=BEST_PROM, distance=BEST_PD, height=BEST_HEIGHT)
        peaks_b, _ = find_peaks(norm_b, prominence=BEST_PROM, distance=BEST_PD, height=BEST_HEIGHT)

        # Plot traces and detected peaks
        for i, (norm_sig, peaks) in enumerate([(norm_a, peaks_a), (norm_b, peaks_b)]):
            ax.plot(norm_sig, color=colors[i], alpha=0.6, label=labels[i] if (row_idx == 0 and col_idx == 0) else "")
            ax.plot(peaks, norm_sig[peaks], "x", color='black', markersize=5, markeredgewidth=1.5)

        N_val = calculate_coincidences(peaks_a, peaks_b, window=BEST_WINDOW)
        counts_dict[f"N({alpha},{beta})"] = N_val

        ax.grid(True, alpha=0.3)
        if row_idx == 0:
            ax.set_title(f"$\\beta = {beta}^\circ$", fontsize=12)
        if col_idx == 3:
            ax2 = ax.twinx()
            ax2.set_ylabel(f"$\\alpha = {alpha}^\circ$", fontsize=12, rotation=270, labelpad=15)
            ax2.set_yticks([])
        if row_idx == 3:
            ax.set_xlabel("Frame")
        if col_idx == 0:
            ax.set_ylabel("Normalized Intensity")

fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95))
plt.subplots_adjust(left=0.05, right=0.92, top=0.95, bottom=0.08, wspace=0.3, hspace=0.3)
plt.savefig("Bell_Inequality_Grid.png", dpi=300)
plt.show()


# ==========================================================
# --- STEP 3: CHSH INEQUALITY EVALUATION ---
# ==========================================================
print("\n--- Final Coincidence Counts ---")
total_counts = 0
for key, value in counts_dict.items():
    print(f"{key}: {value}")
    total_counts += value

abs_S, delta_S = compute_CHSH_S(counts_dict)
z_score = (abs_S - 2.0) / delta_S if (abs_S > 2.0 and delta_S > 0) else 0.0

print("\n==================================")
print(f"Total Coincidences: {total_counts}")
print(f"FINAL S VALUE: {abs_S:.4f} ± {delta_S:.4f}")
print(f"VIOLATION SIGNIFICANCE: {z_score:.2f} Sigma")
print("==================================")