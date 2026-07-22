"""
analysis.py
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.special import digamma
from scipy.optimize import minimize_scalar


# MLE BETA OPTIMIZATION

def _optimize_betas(evidence_matrix, tom_models):
    """Fits the optimal inverse temperature (beta) for each ToM model via MLE."""
    optimized_betas = {}

    for model in tom_models:
        def neg_log_likelihood(beta):
            nll = 0.0
            for game_data in evidence_matrix:
                for move in game_data["evs"][model]:
                    evs = np.array(move["evs"])
                    chosen = move["chosen"]
                    max_ev = np.max(evs)
                    exp_evs = np.exp(beta * (evs - max_ev))
                    probs = exp_evs / np.sum(exp_evs)
                    nll -= np.log(max(probs[chosen], 1e-10))
            return nll

        res = minimize_scalar(neg_log_likelihood, bounds=(0.0, 50.0), method='bounded')
        optimized_betas[model] = res.x
        print(f"Optimized Beta for {model.upper():<5}: {res.x:.4f}")

    return optimized_betas


def _build_likelihood_dataframe(evidence_matrix, shadow_names, optimized_betas):
    """Applies the optimized betas to construct the final log-likelihood matrix."""
    final_evidence_matrix = []

    for game_data in evidence_matrix:
        row = {"game_id": game_data["game_id"]}
        for name in shadow_names:
            game_ll = 0.0
            if name == "random":
                for move in game_data["evs"][name]:
                    game_ll += np.log(max(move["prob"], 1e-10))
            else:
                beta = optimized_betas[name]
                for move in game_data["evs"][name]:
                    evs = np.array(move["evs"])
                    chosen = move["chosen"]
                    max_ev = np.max(evs)
                    exp_evs = np.exp(beta * (evs - max_ev))
                    probs = exp_evs / np.sum(exp_evs)
                    game_ll += np.log(max(probs[chosen], 1e-10))
            row[name] = game_ll
        final_evidence_matrix.append(row)

    return pd.DataFrame(final_evidence_matrix)


# BAYESIAN MODEL SELECTION (BMS)

def run_rfx_bms(L, max_iter=1000, tol=1e-6):
    """
    Runs Random Effects Bayesian Model Selection (Stephan et al., 2009).
    """
    N, K = L.shape
    alpha0 = np.ones(K)
    alpha = np.copy(alpha0)

    for i in range(max_iter):
        E_log_r = digamma(alpha) - digamma(np.sum(alpha))
        log_u = L + E_log_r

        max_log_u = np.max(log_u, axis=1, keepdims=True)
        u = np.exp(log_u - max_log_u)
        g = u / np.sum(u, axis=1, keepdims=True)
        new_alpha = alpha0 + np.sum(g, axis=0)

        if np.linalg.norm(new_alpha - alpha) < tol:
            alpha = new_alpha
            break
        alpha = new_alpha

    exp_r = alpha / np.sum(alpha)

    # Calculate Exceedance Probabilities (XP) via Dirichlet sampling
    n_samples = 100000
    samples = np.random.dirichlet(alpha, n_samples)
    max_idx = np.argmax(samples, axis=1)
    xp = np.array([np.mean(max_idx == k) for k in range(K)])

    return alpha, exp_r, xp


# ANALYSIS PIPELINE

def run_analysis_pipeline(evidence_matrix, shadow_names, csv_path, json_path):
    """
    Master function: Fits betas, builds the likelihood matrix, saves to CSV,
    and immediately runs RFX-BMS on the data in memory.
    """
    print(f"\n{'=' * 80}\nFITTING BETA (MLE) VIA EV CACHING\n{'=' * 80}")

    # Save the raw EV cache
    with open(json_path, "w") as f:
        json.dump(evidence_matrix, f, indent=2)
    print(f"Raw EV cache saved to: {json_path}")

    # Fit Betas
    tom_models = [m for m in shadow_names if m != "random"]
    optimized_betas = _optimize_betas(evidence_matrix, tom_models)

    print("\n" + "=" * 60 + "\n" + " " * 15 + "OPTIMIZED BETAS (MLE)\n" + "=" * 60 + f"\n{'Model':<10} | {'Beta':>16}\n" + "-" * 60)
    for m, b in optimized_betas.items(): print(f"{m.upper():<10} | {b:>16.4f}")
    print("=" * 60)

    # Automatically derive the beta path from the CSV path (same directory)
    dir_evidence = os.path.dirname(csv_path)
    csv_filename = os.path.basename(csv_path)
    beta_filename = csv_filename.replace("evidence_matrix", "betas").replace(".csv", ".json")
    beta_path = os.path.join(dir_evidence, beta_filename)

    with open(beta_path, "w") as f:
        json.dump(optimized_betas, f, indent=2)
    print(f"Optimized betas saved to: {beta_path}")

    # Build Likelihood Matrix
    df = _build_likelihood_dataframe(evidence_matrix, shadow_names, optimized_betas)

    # Save to disk
    df.to_csv(csv_path, index=False)
    print(f"Optimized evidence matrix saved to: {csv_path}")

    # Extract numerical data & Run BMS
    model_names = [col for col in df.columns if col != 'game_id']
    L_matrix = df[model_names].values
    alpha, exp_r, xp = run_rfx_bms(L_matrix)

    # Print Output
    print(f"\nModels detected: {model_names}")
    print(f"Number of games/trials (N): {len(df)}")

    print("\n" + "=" * 60)
    print(" " * 15 + "RFX-BMS RESULTS")
    print("=" * 60)
    print(f"{'Model':<10} | {'Expected Freq (r)':<20} | {'Exceedance Prob (XP)':<20}")
    print("-" * 60)

    for i, name in enumerate(model_names):
        print(f"{name.upper():<10} | {exp_r[i] * 100:>15.2f}%    | {xp[i] * 100:>16.2f}%")
    print("=" * 60)

    return alpha, exp_r, xp


# STANDALONE EXECUTION

if __name__ == "__main__":

    # Simple hardcoded target assuming script is run from the project root
    TARGET_FILE = "raw_evs/raw_ev_cache_claude_vs_tom1_diagnostic.json"

    print(f"Loading data from {TARGET_FILE}...")
    with open(TARGET_FILE, 'r') as f:
        evidence_matrix = json.load(f)

    # Infer shadow agent names from the keys
    shadow_names = list(evidence_matrix[0]["evs"].keys())

    # Dynamically generate output filenames based on the hardcoded target
    base_filename = os.path.basename(TARGET_FILE)
    name_core = base_filename.replace("raw_ev_cache_", "").replace(".json", "")

    # Output path mapping (assuming 'evidence_matrices' is directly inside 'results')
    csv_path = os.path.join("evidence_matrices", f"evidence_matrix_{name_core}.csv")

    # Run the pipeline
    run_analysis_pipeline(evidence_matrix, shadow_names, csv_path, TARGET_FILE)