"""Flask backend API for RL Agent UI.

Serves patient data, agent simulations, and agent metrics.
"""
from __future__ import annotations
from flask import Flask, jsonify, request
from flask_cors import CORS
from pathlib import Path
import numpy as np
import json
import random
from typing import Dict, List, Optional
from datetime import datetime
import sys
import traceback

# Import with error handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available")

try:
    from src.config import load_config, PTV_NAMES, OAR_NAMES, ALL_STRUCTURES
    from src.env.dose_env import DoseEnv
    from src.agents.ppo import PPO
    from src.utils.metrics import d_at_volume
    MODELS_AVAILABLE = True
except Exception as e:
    MODELS_AVAILABLE = False
    print(f"WARNING: Could not import RL models: {e}")
    traceback.print_exc()

app = Flask(__name__)
CORS(app)

# Global state
CONFIG = None
ENV = None
AGENT = None
BACKEND_READY = False
PATIENT_CACHE = {}
EVAL_RESULTS = {}


def init_backend(config_path: str = "configs/default.yaml", 
                 ckpt_path: str = "runs/best.pt",
                 split: str = "validation"):
    """Initialize backend with config, environment, and agent."""
    global CONFIG, ENV, AGENT, BACKEND_READY
    
    try:
        if not MODELS_AVAILABLE:
            print("ERROR: RL models not available, using demo mode")
            CONFIG = {
                'n_fractions': 35,
                'n_beams': 9,
                'beamlet_h': 16,
                'beamlet_w': 16,
                'prescription': {"PTV70": 70.0, "PTV63": 63.0, "PTV56": 56.0},
                'oar_tolerance': {
                    "Brainstem": 54.0, "SpinalCord": 45.0, "Mandible": 70.0,
                    "LeftParotid": 26.0, "RightParotid": 26.0,
                }
            }
            BACKEND_READY = True
            return CONFIG, None, None
        
        CONFIG = load_config(config_path)
        ENV = DoseEnv(CONFIG, split=split)
        
        # Load pretrained agent
        # PPO.__init__ takes (cfg, in_channels)
        # in_channels = CT(1) + structures(3+5) + cumulative_dose(1) + ptv_gap(1) + beam_paths(1) = 12
        in_channels = 12
        AGENT = PPO(CONFIG, in_channels=in_channels)
        
        from src.config import resolve_device
        device = resolve_device(CONFIG.device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        AGENT.net.load_state_dict(checkpoint['net'])
        AGENT.net.eval()
        
        BACKEND_READY = True
        print("✓ Backend initialized successfully")
        return CONFIG, ENV, AGENT
    except Exception as e:
        print(f"ERROR initializing backend: {e}")
        traceback.print_exc()
        BACKEND_READY = True  # Still mark as ready but in degraded mode
        return None, None, None


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok', 
        'timestamp': datetime.now().isoformat(),
        'backend_ready': BACKEND_READY,
        'models_available': MODELS_AVAILABLE
    })


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get system configuration."""
    if CONFIG is None:
        return jsonify({
            'n_fractions': 35,
            'n_beams': 9,
            'beamlet_h': 16,
            'beamlet_w': 16,
            'ptv_names': ["PTV70", "PTV63", "PTV56"],
            'oar_names': ["Brainstem", "SpinalCord", "Mandible", "LeftParotid", "RightParotid"],
            'ptv_prescriptions': {"PTV70": 70.0, "PTV63": 63.0, "PTV56": 56.0},
            'oar_tolerances': {
                "Brainstem": 54.0, "SpinalCord": 45.0, "Mandible": 70.0,
                "LeftParotid": 26.0, "RightParotid": 26.0,
            }
        })
    return jsonify({
        'n_fractions': CONFIG.n_fractions,
        'n_beams': CONFIG.n_beams,
        'beamlet_h': CONFIG.beamlet_h,
        'beamlet_w': CONFIG.beamlet_w,
        'ptv_names': PTV_NAMES if MODELS_AVAILABLE else ["PTV70", "PTV63", "PTV56"],
        'oar_names': OAR_NAMES if MODELS_AVAILABLE else ["Brainstem", "SpinalCord", "Mandible", "LeftParotid", "RightParotid"],
        'ptv_prescriptions': CONFIG.prescription,
        'oar_tolerances': CONFIG.oar_tolerance,
    })


@app.route('/api/patients', methods=['GET'])
def get_patients():
    """Get list of available patients."""
    try:
        if ENV is None:
            # Return demo patients
            demo_patients = [f"pt_{200+i}" for i in range(20)]
            return jsonify({
                'patients': demo_patients,
                'total_count': len(demo_patients),
                'mode': 'demo'
            })
        patients = sorted(ENV.patient_ids)
        return jsonify({
            'patients': patients,
            'total_count': len(patients),
            'mode': 'live'
        })
    except Exception as e:
        print(f"Error getting patients: {e}")
        demo_patients = [f"pt_{200+i}" for i in range(20)]
        return jsonify({
            'patients': demo_patients,
            'total_count': len(demo_patients),
            'mode': 'demo',
            'error': str(e)
        })


@app.route('/api/patients/<patient_id>/data', methods=['GET'])
def get_patient_data(patient_id):
    """Get patient anatomy and dose influence matrix info."""
    try:
        if patient_id not in PATIENT_CACHE:
            # Load patient data
            state, _ = ENV.reset(patient_id=patient_id)
            patient_data = {
                'patient_id': patient_id,
                'state_shape': list(state.shape),
                'structures': {
                    'ptvs': PTV_NAMES,
                    'oars': OAR_NAMES
                }
            }
            PATIENT_CACHE[patient_id] = patient_data
        
        return jsonify(PATIENT_CACHE[patient_id])
    except Exception as e:
        return jsonify({'error': str(e)}), 400


def _present_structure_names():
    """Canonical structures that are actually contoured for the *current*
    ENV patient (mask exists and is non-empty).

    Returns ``None`` when the env / masks are unavailable (e.g. demo mode),
    in which case callers should treat every structure as present.
    """
    if ENV is None or not MODELS_AVAILABLE:
        return None
    try:
        masks = ENV._structure_masks()
    except Exception as e:
        print(f"Could not determine present structures: {e}")
        return None
    return {name for name, mask in masks.items() if np.any(mask)}


def _structures_payload(present_structures, ptv_names, oar_names):
    """Split the canonical structure lists into present / missing for the UI."""
    if present_structures is None:
        return {
            'ptv_present': list(ptv_names), 'ptv_missing': [],
            'oar_present': list(oar_names), 'oar_missing': [],
        }
    return {
        'ptv_present': [p for p in ptv_names if p in present_structures],
        'ptv_missing': [p for p in ptv_names if p not in present_structures],
        'oar_present': [o for o in oar_names if o in present_structures],
        'oar_missing': [o for o in oar_names if o not in present_structures],
    }


def _build_summary(fraction_data, prescriptions, tolerances,
                   present_structures=None, ptv_d95=None):
    """Build end-of-treatment summary from all fraction data.

    ``present_structures`` is the set of structures actually contoured for
    this patient (``None`` => assume all present, e.g. demo mode). Missing
    structures are flagged with ``present: False`` and carry no dose /
    coverage / violation numbers so the UI can distinguish "not contoured"
    from "contoured but received 0 Gy".

    ``ptv_d95`` maps each PTV to its **D95** (the dose received by at least
    95% of the PTV volume) -- the clinical coverage metric, identical to the
    ``D95_<PTV>`` column in ``evaluate.py``. When provided, PTV coverage is
    reported against D95 (so the UI agrees with ``evaluate.py``); the mean
    dose is still carried as ``final_dose`` for reference. When absent (no
    real dose volume), coverage falls back to the mean dose.
    """
    if not fraction_data:
        return {}
    last = fraction_data[-1]
    final_doses = last.get('cumulative_organ_doses', {})
    total_reward = sum(f.get('reward', 0) for f in fraction_data)
    avg_reward = total_reward / len(fraction_data)
    ptv_d95 = ptv_d95 or {}

    def _is_present(name):
        return present_structures is None or name in present_structures

    ptv_summary = {}
    for ptv, rx in prescriptions.items():
        present = _is_present(ptv)
        mean_dose = final_doses.get(ptv, 0)
        d95 = ptv_d95.get(ptv)
        has_d95 = present and d95 is not None
        # Coverage is D95-based when available (matches evaluate.py); otherwise
        # falls back to the mean dose. Uncapped so over-coverage (>100%) shows.
        coverage_basis = d95 if has_d95 else mean_dose
        coverage_pct = (coverage_basis / rx) * 100 if rx > 0 else 0
        achieved = (d95 >= rx) if has_d95 else (coverage_pct >= 95.0)
        ptv_summary[ptv] = {
            'present': present,
            'final_dose': round(mean_dose, 2) if present else None,   # mean dose
            'd95': round(d95, 2) if has_d95 else None,
            'prescribed': rx,
            'coverage_pct': round(coverage_pct, 1) if present else None,
            'coverage_metric': 'D95' if has_d95 else 'mean',
            'achieved': bool(present and achieved),
        }

    oar_summary = {}
    for oar, tol in tolerances.items():
        present = _is_present(oar)
        dose = final_doses.get(oar, 0)
        violation = bool(present and dose > tol)
        oar_summary[oar] = {
            'present': present,
            'final_dose': round(dose, 2) if present else None,
            'tolerance': tol,
            'pct_of_tolerance': (round((dose / tol) * 100, 1) if tol > 0 else 0)
                                if present else None,
            'violation': violation,
        }

    return {
        'total_reward': round(total_reward, 4),
        'avg_reward_per_fraction': round(avg_reward, 4),
        'ptv_summary': ptv_summary,
        'oar_summary': oar_summary,
        'n_fractions': len(fraction_data),
    }


def _dvh_from_dose(cumulative_dose, structure_masks, present_structures,
                   n_points: int = 60, max_dose: float = 80.0):
    """Cumulative dose-volume histogram per *present* structure.

    For each structure returns ``{dose: [...Gy], volume: [...fraction in [0,1]]}``
    where ``volume[i]`` is the fraction of the structure's voxels receiving at
    least ``dose[i]`` Gy. Downsampled to ``n_points`` so the JSON stays small.
    """
    dose_axis = np.linspace(0.0, max_dose, n_points)
    dvh = {}
    for name, mask in structure_masks.items():
        if present_structures is not None and name not in present_structures:
            continue
        if mask is None or mask.sum() == 0:
            continue
        dose_in_mask = cumulative_dose[mask > 0]
        volume = [float((dose_in_mask >= d).mean()) for d in dose_axis]
        dvh[name] = {
            'dose': [round(float(d), 2) for d in dose_axis],
            'volume': [round(v, 4) for v in volume],
        }
    return dvh


def _dvh_demo(cumulative_organ_doses, prescriptions,
              n_points: int = 60, max_dose: float = 80.0):
    """Synthetic but plausible DVH curves for demo mode.

    Models each structure's voxel doses as a clipped normal around its mean
    dose (tight spread for PTVs -> near-step at prescription, broad for OARs).
    """
    dose_axis = np.linspace(0.0, max_dose, n_points)
    rng = np.random.default_rng(0)
    dvh = {}
    for name, mean_dose in cumulative_organ_doses.items():
        is_ptv = name in prescriptions
        spread = max(mean_dose * (0.06 if is_ptv else 0.4), 1.0)
        samples = np.clip(rng.normal(mean_dose, spread, 4000), 0.0, None)
        volume = [float((samples >= d).mean()) for d in dose_axis]
        dvh[name] = {
            'dose': [round(float(d), 2) for d in dose_axis],
            'volume': [round(v, 4) for v in volume],
        }
    return dvh


@app.route('/api/patients/<patient_id>/simulate', methods=['POST'])
def simulate_patient(patient_id):
    """Run agent simulation on a patient for all fractions."""
    try:
        prescriptions = CONFIG.prescription if CONFIG else {"PTV70": 70.0, "PTV63": 63.0, "PTV56": 56.0}
        tolerances = CONFIG.oar_tolerance if CONFIG else {
            "Brainstem": 54.0, "SpinalCord": 45.0, "Mandible": 70.0,
            "LeftParotid": 26.0, "RightParotid": 26.0,
        }
        n_fractions = CONFIG.n_fractions if CONFIG else 35
        lambda_oar = float(CONFIG.lambda_oar) if CONFIG else 1.0
        lambda_ptv = float(CONFIG.lambda_ptv) if CONFIG else 1.0

        if ENV is None:
            # Demo mode - simulate all 35 fractions realistically
            fraction_data = []
            cumulative = {k: 0.0 for k in list(prescriptions.keys()) + list(tolerances.keys())}
            for i in range(n_fractions):
                beam_heatmap = []
                for b in range(9):
                    beam_doses = (0.3 + i * 0.005) * np.random.rand(16, 16)
                    beam_heatmap.append(beam_doses.tolist())
                # Increment cumulative doses per fraction
                per_frac = {
                    'PTV70': 70.0 / n_fractions * (0.9 + 0.2 * np.random.rand()),
                    'PTV63': 63.0 / n_fractions * (0.9 + 0.2 * np.random.rand()),
                    'PTV56': 56.0 / n_fractions * (0.9 + 0.2 * np.random.rand()),
                    'Brainstem': 54.0 / n_fractions * 0.4 * np.random.rand(),
                    'SpinalCord': 45.0 / n_fractions * 0.35 * np.random.rand(),
                    'Mandible': 70.0 / n_fractions * 0.45 * np.random.rand(),
                    'LeftParotid': 26.0 / n_fractions * 0.5 * np.random.rand(),
                    'RightParotid': 26.0 / n_fractions * 0.5 * np.random.rand(),
                }
                for k in cumulative:
                    cumulative[k] += per_frac.get(k, 0)
                # Synthetic per-fraction D95: with the demo PTV spread (~6% of
                # mean) the 5th-percentile dose sits ~0.9x the mean. Grows each
                # fraction so the UI's PTV buckets fill up over the course.
                ptv_d95_frac = {ptv: round(cumulative[ptv] * 0.9, 2)
                                for ptv in prescriptions}
                fraction_data.append({
                    'fraction': i + 1,
                    'reward': float(0.5 + (i * 0.005) + (np.random.random() * 0.05)),
                    'oar_penalty': float(max(0, 0.08 - (i * 0.001))),
                    'ptv_reward': float(0.6 + (i * 0.003)),
                    'action_mean': float(0.3 + np.random.random() * 0.1),
                    'action_max': float(0.7 + np.random.random() * 0.2),
                    'beam_heatmap': beam_heatmap,
                    'cumulative_organ_doses': {k: round(v, 2) for k, v in cumulative.items()},
                    'ptv_d95': ptv_d95_frac,
                    'lambda_oar': lambda_oar,
                    'lambda_ptv': lambda_ptv,
                })
            # Demo mode: all structures are synthetic and therefore present.
            present_structures = None
            dvh = _dvh_demo(fraction_data[-1]['cumulative_organ_doses'],
                            prescriptions)
            summary = _build_summary(fraction_data, prescriptions, tolerances,
                                     present_structures,
                                     fraction_data[-1]['ptv_d95'])
            return jsonify({
                'patient_id': patient_id,
                'fractions': fraction_data,
                'total_fractions_simulated': len(fraction_data),
                'summary': summary,
                'dvh': dvh,
                'structures': _structures_payload(
                    present_structures,
                    list(prescriptions.keys()), list(tolerances.keys()),
                ),
                'mode': 'demo',
            })

        # The patient data loaded but the trained agent did not. Rather than
        # run a random policy and present its output as if it were a real plan,
        # refuse: the UI surfaces this error and shows no numbers at all.
        if AGENT is None:
            return jsonify({
                'error': 'Trained agent unavailable (checkpoint not loaded). '
                         'Refusing to return a random-policy plan.',
                'mode': 'agent-unavailable',
            }), 503

        # Live mode - run all n_fractions
        state, fraction_progress = ENV.reset(patient_id=patient_id)
        patient_done = False
        fraction_data = []
        fraction_idx = 0

        fraction_ptv_d95 = {}
        while not patient_done and fraction_idx < n_fractions:
            action, _, _, _ = AGENT.act(state, fraction_progress, deterministic=True)

            action_2d = action.reshape((CONFIG.n_beams, CONFIG.beamlet_h, CONFIG.beamlet_w))
            beam_heatmap = [action_2d[b].tolist() for b in range(CONFIG.n_beams)]

            state, fraction_progress, reward, done, info = ENV.step(action)
            patient_done = info.get('patient_done', done)

            # Dose stats are taken AFTER the step, so each fraction reflects the
            # cumulative dose delivered *through* that fraction (it grows fraction
            # by fraction). cumulative_organ_doses = mean dose per structure;
            # ptv_d95 = the per-fraction D95 (dose to 95% of the PTV) so the UI's
            # PTV buckets fill up over the course and land on evaluate.py's value
            # at the final fraction.
            structure_masks = ENV._structure_masks()
            cumulative_organ_doses = {}
            for struct_name, mask in structure_masks.items():
                mean_dose = float(np.mean(ENV.cumulative_dose[mask > 0])) if np.any(mask) else 0.0
                cumulative_organ_doses[struct_name] = round(mean_dose, 2)
            fraction_ptv_d95 = {}
            for ptv in prescriptions:
                mask = structure_masks.get(ptv)
                if mask is not None and np.any(mask):
                    fraction_ptv_d95[ptv] = round(
                        float(d_at_volume(ENV.cumulative_dose, mask, 0.95)), 2
                    )

            fraction_data.append({
                'fraction': info.get('fraction_index', fraction_idx + 1),
                'reward': float(reward),
                'oar_penalty': float(info.get('oar_penalty', 0)),
                'ptv_reward': float(info.get('ptv_reward', 0)),
                'action_mean': float(action.mean()),
                'action_max': float(action.max()),
                'beam_heatmap': beam_heatmap,
                'cumulative_organ_doses': cumulative_organ_doses,
                'ptv_d95': dict(fraction_ptv_d95),
                'lambda_oar': float(info.get('lambda_oar', lambda_oar)),
                'lambda_ptv': float(info.get('lambda_ptv', lambda_ptv)),
            })
            fraction_idx += 1

        # Which canonical structures are actually contoured for this patient.
        present_structures = _present_structure_names()
        structure_masks = ENV._structure_masks()
        # End-of-course D95 = the final fraction's D95 (full 35-fraction dose),
        # identical to the D95_<PTV> column in evaluate.py.
        summary = _build_summary(fraction_data, prescriptions, tolerances,
                                 present_structures, fraction_ptv_d95)
        # Cumulative DVH over the finished course (real dose vs structure masks).
        dvh = _dvh_from_dose(ENV.cumulative_dose, structure_masks,
                             present_structures)
        return jsonify({
            'patient_id': patient_id,
            'fractions': fraction_data,
            'total_fractions_simulated': len(fraction_data),
            'summary': summary,
            'dvh': dvh,
            'structures': _structures_payload(
                present_structures,
                list(prescriptions.keys()), list(tolerances.keys()),
            ),
            'mode': 'live',
        })
    except Exception as e:
        print(f"Error in simulate_patient: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 400


@app.route('/api/patients/random', methods=['GET'])
def get_random_patient():
    """Get a random patient ID."""
    patient_id = random.choice(ENV.patient_ids)
    return jsonify({'patient_id': patient_id})


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current system status and metrics."""
    return jsonify({
        'total_patients': len(ENV.patient_ids),
        'n_fractions': CONFIG.n_fractions,
        'agent_ready': AGENT is not None,
        'config_loaded': CONFIG is not None,
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--ckpt', default='runs/best.pt')
    parser.add_argument('--split', default='validation')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    
    print("Initializing backend...")
    init_backend(args.config, args.ckpt, args.split)
    print("Backend ready!")
    
    app.run(debug=args.debug, port=args.port, host='0.0.0.0')
