# Experimental Verification of Bell's Inequality Violation and Quantum State Tomography via Photonic Polarization Measurements

## Abstract
This repository contains the computational pipeline for experimental quantum optics measurements of polarization-entangled photon pairs. The paper investigates the non-local properties of entangled photons by testing the Clauser-Horne-Shimony-Holt (CHSH) form of Bell's inequality and performing quantum state tomography (QST) to reconstruct two-qubit density matrices (ρ). The provided Python scripts process raw experimental video recordings of photodetector signals, apply color-masking and peak-detection algorithms to extract coincidences, and simulate theoretical state reconstructions. Executing these scripts yields a 4×4 visualization grid of time-series photon pulses with detected coincidences, calculated CHSH violation values (S ± ΔS), and 3D bar plots representing reconstructed quantum density matrices.

---

## File Descriptions

### 1. `bell_inequality_chsh.py`
* **Description:** Performs automated analysis on video recordings of polarization-entangled photon measurements across 16 combinations of Alice's (α ∈ {-45°, 0°, 45°, 90°}) and Bob's (β ∈ {-22.5°, 22.5°, 67.5°, 112.5°}) polarizer angles. It extracts red-channel HSV intensity signals from Region of Interests (ROIs), applies temporal channel shifting to align signals, detects intensity peaks, and calculates two-fold coincidence counts within a specified time window. Finally, it evaluates the CHSH inequality parameter S, its statistical uncertainty ΔS, and the statistical significance (z-score) of local realism violation.
* **Inputs:** 
  * 16 video files (`.mp4`) located in structured subdirectories matching the polarizer configurations: `./ALPHA {alpha}/{beta}.mp4` (e.g., `./ALPHA 0/22.5.mp4`).
* **Outputs:** 
  * `Bell_Inequality_Grid.png`: A high-resolution 4×4 grid figure showing normalized intensity time-series traces and detected peaks (x) for Alice and Bob across all angle settings.
  * Console printouts displaying raw coincidence counts N(α, β), total combined coincidences, the final calculated value of S ± ΔS, and the violation significance in standard deviations (σ).

---

### 2. `quantum_state_tomography.py`
* **Description:** Combines a theoretical simulation module and an experimental video-analysis framework for two-qubit Quantum State Tomography (QST). The script simulates theoretical density matrices (ρ) for Bell states (|Φ⁺⟩, |Ψ⁺⟩) and mixed states, calculates expected projection populations, and reconstructs density matrices from population counts. For experimental video analysis, it provides a user-interactive ROI selector, extracts multi-channel red-intensity signals for detector paths (VV, HH, VH, HV), identifies photon pulse events using height/distance thresholds, calculates coincidence matrices, and manages random bit-sequence data in Excel spreadsheets.
* **Inputs:** 
  * Experimental MP4 video files (e.g., `part1-10.mp4`).
  * Interactive user bounding-box selections drawn on the OpenCV GUI for sensor ROIs.
* **Outputs:** 
  * `part one bit.xlsx`: An Excel spreadsheet containing generated random bit sequences (10-bit, 25-bit, and 50-bit).
  * `measurments and peaks - bits.png`: A 4-panel subplot showing raw intensity channels, applied detection thresholds, and identified pulse peaks.
  * Interactive 3D bar chart windows displaying the real part of reconstructed density matrices (ρ) for both theoretical simulations and experimental state measurements.
  * Console output detailing coincidence counts (N_HH, N_HV, N_VH, N_VV) and matrix elements.
