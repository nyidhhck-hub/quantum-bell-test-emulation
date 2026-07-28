import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.signal import find_peaks


# ==========================================
# --- 1. THEORETICAL SIMULATION SECTION ---
# ==========================================

def build_ideal_state(state_name="state_A"):
    """Returns theoretical state for simulation."""
    if state_name == "state_A":
        wavefunc = np.array([1, 0, 0, 1]) / np.sqrt(2)
        return np.outer(wavefunc, wavefunc)
    elif state_name == "state_B":
        wavefunc = np.array([0, 1, 1, 0]) / np.sqrt(2)
        return np.outer(wavefunc, wavefunc)
    elif state_name == "mixed_state":
        return np.diag([0.5, 0, 0, 0.5])
    else:
        raise ValueError("Unknown quantum state")


def calculate_expected_counts(rho_matrix, angle_alice_deg, angle_bob_deg, total_pulses=1000):
    """Simulates photon counts for specific polarizer angles."""
    rad_a = np.radians(angle_alice_deg)
    rad_b = np.radians(angle_bob_deg)

    vec_alice = np.array([np.cos(rad_a), np.sin(rad_a)])
    vec_bob = np.array([np.cos(rad_b), np.sin(rad_b)])

    operator = np.kron(np.outer(vec_alice, vec_alice), np.outer(vec_bob, vec_bob))
    probability = np.trace(rho_matrix @ operator)
    probability = np.clip(np.real(probability), 0, 1)

    return int(np.round(probability * total_pulses))


def reconstruct_from_counts(cnt_hh, cnt_hv, cnt_vh, cnt_vv):
    """Reconstructs the matrix using simulated population data."""
    total_events = cnt_vv + cnt_hh + cnt_vh + cnt_hv
    if total_events == 0:
        total_events = 1

    rho_diag = [cnt_hh / total_events, cnt_hv / total_events,
                cnt_vh / total_events, cnt_vv / total_events]

    rho_rec = np.diag(rho_diag)

    if (rho_diag[0] + rho_diag[3]) > (rho_diag[1] + rho_diag[2]):
        print("Detected Correlated State (Simulation)")
        coh = np.sqrt(rho_diag[0] * rho_diag[3])
        rho_rec[0, 3] = rho_rec[3, 0] = coh
    else:
        print("Detected Anti-Correlated State (Simulation)")
        coh = np.sqrt(rho_diag[1] * rho_diag[2])
        rho_rec[1, 2] = rho_rec[2, 1] = coh

    return rho_rec


# ==========================================
# --- 2. EXPERIMENTAL VIDEO SECTION ---
# ==========================================

class PhotonicMeasurementSystem:
    def __init__(self, vid_source, shrink_val=0.5, swap_bob=False):
        self.vid_source = vid_source
        self.shrink_val = shrink_val
        self.swap_bob = swap_bob

        self.sensor_tags = ["Alice V", "Alice H", "Bob V", "Bob H"]
        self.bounding_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
        self.intensity_limits = [30000.0, 50000.0, 100000.0, 60000.0]

        if self.swap_bob:
            self.intensity_limits[2], self.intensity_limits[3] = self.intensity_limits[3], self.intensity_limits[2]

    def manage_bit_files(self, out_file="part one bit.xlsx"):
        seq_dict = {
            "10 bit random list": [int(b) for b in "1001101101"],
            "25 bit random list": [int(b) for b in "1100101101001110101001101"],
            "50 bit random list": [int(b) for b in "01101001110010110100110101100100111010100101100110"]
        }

        try:
            with pd.ExcelWriter(out_file, engine='openpyxl') as wb_writer:
                for sheet_title, bits_array in seq_dict.items():
                    pd.DataFrame(bits_array).to_excel(wb_writer, sheet_name=sheet_title, index=False, header=False)
            print(f"Success! '{out_file}' was created perfectly.")
        except ModuleNotFoundError:
            print("Error: Please install openpyxl.")

        for key_name in seq_dict.keys():
            if key_name[:2] in self.vid_source:
                return seq_dict[key_name]
        return seq_dict["10 bit random list"]

    def extract_red_channel(self):
        if not os.path.exists(self.vid_source):
            raise FileNotFoundError("Error: Video file not found.")

        stream = cv2.VideoCapture(self.vid_source)
        success, initial_frame = stream.read()
        if not success:
            raise ValueError("Could not read the video.")

        mini_view = cv2.resize(initial_frame, (0, 0), fx=self.shrink_val, fy=self.shrink_val)
        enlarge_ratio = 1 / self.shrink_val
        selected_zones = []

        for idx, tag in enumerate(self.sensor_tags):
            win_title = f"Select {tag} (Draw & Press SPACE)"
            box = cv2.selectROI(win_title, mini_view, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow(win_title)
            cv2.waitKey(1)

            if box == (0, 0, 0, 0):
                box = (0, 0, 1, 1)

            scaled_coords = (int(box[0] * enlarge_ratio), int(box[1] * enlarge_ratio),
                             int(box[2] * enlarge_ratio), int(box[3] * enlarge_ratio))
            selected_zones.append(scaled_coords)

            cv2.rectangle(mini_view, (box[0], box[1]),
                          (box[0] + box[2], box[1] + box[3]), self.bounding_colors[idx], 2)

        stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
        intensity_streams = [[] for _ in range(4)]

        while True:
            success, current_frame = stream.read()
            if not success: break

            for idx, (pos_x, pos_y, width, height) in enumerate(selected_zones):
                cropped_area = current_frame[pos_y:pos_y + height, pos_x:pos_x + width]
                hsv_area = cv2.cvtColor(cropped_area, cv2.COLOR_BGR2HSV)

                lower_red = cv2.inRange(hsv_area, np.array([0, 100, 100]), np.array([10, 255, 255]))
                upper_red = cv2.inRange(hsv_area, np.array([170, 100, 100]), np.array([180, 255, 255]))
                intensity_streams[idx].append(np.sum(lower_red | upper_red))

        stream.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        if self.swap_bob:
            intensity_streams[2], intensity_streams[3] = intensity_streams[3], intensity_streams[2]

        return intensity_streams

    def identify_pulses(self, raw_data, min_gap=10):
        event_frames = []
        plot_tints = ["b", "g", "r", "orange"]

        plt.figure(figsize=(14, 12))
        print(f"\n{'ROI':<10} | {'Count':<5} | {'Threshold':<10}")
        print("-" * 30)

        for idx in range(4):
            data_arr = np.array(raw_data[idx])
            pulse_locs, _ = find_peaks(data_arr, height=self.intensity_limits[idx], distance=min_gap)
            event_frames.append(pulse_locs)

            print(f"{self.sensor_tags[idx]:<10} | {len(pulse_locs):<5} | {self.intensity_limits[idx]:<10}")

            plt.subplot(4, 1, idx + 1)
            plt.plot(data_arr, color=plot_tints[idx], alpha=0.6)
            plt.plot(pulse_locs, data_arr[pulse_locs], "x", color="black", label="Detected Pulse")
            plt.axhline(self.intensity_limits[idx], color="k", linestyle="--", alpha=0.5, label="Threshold")
            plt.ylabel("Intensity")
            plt.title(f"{self.sensor_tags[idx]} - bits")
            plt.legend(loc="upper left")
            plt.grid(True, alpha=0.3)

        plt.xlabel("Frame Number")
        plt.tight_layout()

        # --- הוספת שמירת התמונה ---
        plt.savefig("measurments and peaks -  bits")

        plt.show()

        return event_frames

    def calculate_coincidences_and_matrix(self, event_frames, time_window=10):
        def match_events(peaks_a, peaks_b):
            return sum(1 for pa in peaks_a if np.any(np.abs(peaks_b - pa) <= time_window))

        coinc_dict = {
            'VV': match_events(event_frames[0], event_frames[2]),
            'HH': match_events(event_frames[1], event_frames[3]),
            'VH': match_events(event_frames[0], event_frames[3]),
            'HV': match_events(event_frames[1], event_frames[2])
        }

        print("\n--- COINCIDENCE RESULTS (EXPERIMENTAL) ---")
        for st, amount in coinc_dict.items():
            print(f"N_{st}: {amount}")

        sum_events = sum(coinc_dict.values()) or 1
        density_mat = np.zeros((4, 4))

        density_mat[0, 0] = coinc_dict['HH'] / sum_events
        density_mat[1, 1] = coinc_dict['HV'] / sum_events
        density_mat[2, 2] = coinc_dict['VH'] / sum_events
        density_mat[3, 3] = coinc_dict['VV'] / sum_events

        if (density_mat[0, 0] + density_mat[3, 3]) > (density_mat[1, 1] + density_mat[2, 2]):
            coherence_val = np.sqrt(density_mat[0, 0] * density_mat[3, 3])
            density_mat[0, 3] = density_mat[3, 0] = coherence_val
        else:
            coherence_val = np.sqrt(density_mat[1, 1] * density_mat[2, 2])
            density_mat[1, 2] = density_mat[2, 1] = coherence_val

        return density_mat


# ==========================================
# --- 3. PLOTTING FUNCTION ---
# ==========================================

def display_tomography_3d(matrix_data, chart_heading):
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.view_init(elev=30, azim=-37.5)

    x_labels = ["HH", "HV", "VH", "VV"]
    y_labels = ["HH", "HV", "VH", "VV"]
    _x = np.arange(4)
    _y = np.arange(4)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    top = np.real(matrix_data).ravel()
    bottom = np.zeros_like(top)
    width = depth = 0.5

    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=0, vmax=0.8)
    bar_colors = [cmap(norm(h)) for h in top]

    ax.bar3d(x, y, bottom, width, depth, top, color=bar_colors, edgecolor='black', linewidth=0.6, shade=True)

    ax.set_xticks(_x + 0.25)
    ax.set_xticklabels(x_labels, fontweight="bold")
    ax.set_yticks(_y + 0.25)
    ax.set_yticklabels(y_labels, fontweight="bold")
    ax.set_zlim(0, 0.8)

    ax.set_xlabel("Detector", fontweight="bold", labelpad=12, fontsize=11)
    ax.set_ylabel("Source", fontweight="bold", labelpad=12, fontsize=11)

    ax.set_title(chart_heading, fontsize=14, fontweight="bold", pad=20)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cb = fig.colorbar(mappable, ax=ax, shrink=0.7, aspect=18, pad=0.08)
    cb.ax.tick_params(labelsize=10)

    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


# ==========================================
# --- MAIN EXECUTION ---
# ==========================================

if __name__ == "__main__":

    # ---------------------------------------------------------
    # שלב א': יצירת המספרים האקראיים
    # ---------------------------------------------------------
    num_random_bits = 50
    generated_array = np.random.randint(2, size=num_random_bits)
    print("--- Generated Random Array ---")
    print(generated_array)
    print("\n")

    # ---------------------------------------------------------
    # שלב ב': סימולציה תיאורטית מלאה
    # ---------------------------------------------------------
    print("--- Generating Simulated Data ---")
    target_sim_state = "state_B"
    ideal_rho = build_ideal_state(target_sim_state)
    sim_pulses = 100

    c_hh = calculate_expected_counts(ideal_rho, 0, 0, sim_pulses)
    c_hv = calculate_expected_counts(ideal_rho, 0, 90, sim_pulses)
    c_vh = calculate_expected_counts(ideal_rho, 90, 0, sim_pulses)
    c_vv = calculate_expected_counts(ideal_rho, 90, 90, sim_pulses)

    print(f"Counts: HH={c_hh}, HV={c_hv}, VH={c_vh}, VV={c_vv}")

    print("\n--- Reconstructing Matrix (Theoretical) ---")
    simulated_matrix = reconstruct_from_counts(c_hh, c_hv, c_vh, c_vv)
    print("\nReconstructed Matrix (Real Part):")
    print(np.round(np.real(simulated_matrix), 3))

    print("\n--- Plotting Simulation ---")
    if target_sim_state == "state_A":
        sim_title = f"Reconstructed Simulated Density Matrix - HHVV {sim_pulses} bits"
    else:
        sim_title = f"Reconstructed Simulated Density Matrix - HVVH {sim_pulses} bits"

    display_tomography_3d(simulated_matrix, chart_heading=sim_title)

    # ---------------------------------------------------------
    # שלב ג': ניתוח הוידאו הניסויי
    # ---------------------------------------------------------
    print("\n\n--- Starting Experimental Video Analysis ---")
    analyzer_sys = PhotonicMeasurementSystem(vid_source=r"part1-10.mp4", swap_bob=False)

    # הפקת רצפים
    target_sequence = analyzer_sys.manage_bit_files()
    sequence_length = len(target_sequence)

    # חילוץ עוצמות אדומות מהוידאו
    print(f"Please select {len(analyzer_sys.sensor_tags)} different areas.")
    intensity_arrays = analyzer_sys.extract_red_channel()

    # איתור פולסים בגרף
    found_pulses = analyzer_sys.identify_pulses(intensity_arrays)

    # חישוב מטריצה סופית
    reconstructed_mat = analyzer_sys.calculate_coincidences_and_matrix(found_pulses)

    # הצגה תלת ממדית
    if analyzer_sys.swap_bob:
        plot_heading = f"Reconstructed Density Matrix - HVVH {sequence_length} bits"
    else:
        plot_heading = f"Reconstructed Density Matrix - HHVV {sequence_length} bits"

    display_tomography_3d(reconstructed_mat, chart_heading=plot_heading)