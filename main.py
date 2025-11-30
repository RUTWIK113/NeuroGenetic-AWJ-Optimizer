import os
import pandas as pd
from control.ga_optimizer import run_genetic_algorithm
from vision.monitoring import measure_nozzle_diameter
from verify_params import verify_parameters_with_llm


from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

print("API Key loaded:", api_key)

# --- 1. CONFIGURATION & RANGES ---

# These names must match the columns in your training data
OPTIMIZABLE_PARAMS = [
    "P (MPa)",
    "mf (kg/min)",
    "v (mm/min)"
]

STATIC_PARAMS = [
    "df (mm)",  # Focusing Nozzle
    "do (mm)"  # Orifice
]

# Safe Operating Ranges for the Machine
PARAM_RANGES = {
    "P (MPa)": [100.0, 400.0],
    "mf (kg/min)": [0.1, 1.0],
    "v (mm/min)": [100.0, 5000.0],
    "df (mm)": [0.76, 1.6],
    "do (mm)": [0.1, 0.3]
}

# Vision System Calibration
PIXELS_TO_MM_RATIO = 0.01


# --------------------------------------------------

def main_control_loop():
    """
    Main loop for the 5-Input Adaptive AWJ Control System.
    """
    print("\n===============================================")
    print("   INTELLIGENT AWJ ADAPTIVE CONTROL SYSTEM     ")
    print("===============================================\n")

    # --- PHASE 1: USER INPUT ---
    try:
        desired_depth = float(input(">> Enter Desired Depth of Cut (mm): "))
    except ValueError:
        print("Invalid input. Defaulting to 10.0 mm.")
        desired_depth = 10.0

    # --- PHASE 2: MACHINE MONITORING ---
    print("\n[PHASE 2] Machine Status Monitoring")
    print("-" * 35)

    # A. Focusing Tube Diameter (df) - Option for Vision or Manual
    use_vision = input(">> Use Vision System to measure Nozzle Wear (df)? (y/n): ").strip().lower()

    df_dia = 0.72  # Default value if everything fails

    if use_vision == 'y':
        default_img = os.path.join('vision', 'test_images', 'nozzle_tip.jpg')
        img_path = input(f">> Enter image path (Press Enter for '{default_img}'): ").strip()

        if not img_path:
            img_path = default_img

        print(f"   Analyzing image: {img_path}...")

        # Call the Computer Vision Module
        inner_d, outer_d = measure_nozzle_diameter(img_path, PIXELS_TO_MM_RATIO)

        if inner_d is not None:
            print(f"   ✅ SUCCESS: Vision detected Focusing Tube (df) = {inner_d:.4f} mm")
            df_dia = inner_d
        else:
            print("   ❌ FAILURE: Vision system could not detect nozzle. Switching to manual.")
            try:
                df_dia = float(input(f"   >> Enter Focusing Nozzle Diameter manually (df, mm): "))
            except ValueError:
                print("   Invalid input. Using default 0.72 mm")
    else:
        # Manual Input Mode
        try:
            df_dia = float(input(f"   >> Enter Focusing Nozzle Diameter (df, mm): "))
        except ValueError:
            print("   Invalid input. Using default 0.72 mm")

    # B. Orifice Diameter (do) - Manual Entry
    try:
        do_dia = float(input(f"   >> Enter Orifice Diameter (do, mm): "))
    except ValueError:
        print("   Invalid input. Using default 0.24 mm")
        do_dia = 0.24

    # Pack static inputs [df, do]
    static_inputs = [df_dia, do_dia]

    # --- PHASE 3: OPTIMIZATION (GA) ---
    print("\n[PHASE 3] Genetic Optimization")
    print("-" * 35)
    print(f"Target: {desired_depth} mm | State: df={df_dia}mm, do={do_dia}mm")

    # Configure ranges for the optimizer
    optimizable_params_config = {
        name: PARAM_RANGES[name] for name in OPTIMIZABLE_PARAMS
    }

    # Run the GA
    optimal_params = run_genetic_algorithm(
        param_ranges=optimizable_params_config,
        static_inputs=static_inputs,
        desired_depth=desired_depth
    )

    # --- PHASE 4: AI VERIFICATION (LLM) ---
    print("\n[PHASE 4] AI Safety Verification")
    print("-" * 35)

    opt_P = optimal_params['pressure']
    opt_mf = optimal_params['flow_rate']
    opt_v = optimal_params['traverse_rate']

    # Call Gemini Agent
    verification_json = verify_parameters_with_llm(
        opt_P, opt_mf, opt_v,
        df_dia, do_dia, desired_depth
    )

    # Parse Verification Result
    is_safe = False
    if verification_json:
        if isinstance(verification_json, dict) and "error" in verification_json:
            print(f"   Warning: {verification_json['error']}")
        else:
            verdict = verification_json.get('verdict', 'UNKNOWN')
            confidence = verification_json.get('confidence', 'UNKNOWN')
            reasoning = verification_json.get('reasoning', 'No details.')

            print(f"   Verdict:    {verdict}")
            print(f"   Confidence: {confidence}")
            print(f"   Reasoning:  {reasoning}")

            if verdict == "SAFE":
                is_safe = True
    else:
        print("   LLM verification failed.")

    # --- PHASE 5: EXECUTION ---
    print("\n[PHASE 5] Final Execution")
    print("=" * 35)

    if is_safe:
        print("✅ PARAMETERS APPROVED. SENDING TO CONTROLLER...")
        print(f"   [P]  Pressure:      {opt_P:.2f} MPa")
        print(f"   [mf] Abrasive Flow: {opt_mf:.3f} kg/min")
        print(f"   [v]  Traverse Rate: {opt_v:.2f} mm/min")
    else:
        print("⚠️ PARAMETERS FLAGGED AS UNSAFE/UNCERTAIN.")
        print("   Human intervention required before execution.")
        print(f"   Suggested: P={opt_P:.2f}, mf={opt_mf:.3f}, v={opt_v:.2f}")

    print("===============================================\n")


# Run the system
if __name__ == "__main__":
    main_control_loop()