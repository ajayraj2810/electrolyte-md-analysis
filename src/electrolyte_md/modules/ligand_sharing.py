from __future__ import annotations
import argparse
import json
import logging
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple
import numpy as np
import pandas as pd
DEFAULT_OUTPUT_DIR = Path('08_ligand_multimetal_coordination')
DEFAULT_CHUNK_SIZE = 200000
DEFAULT_BLOCK_NS = 10.0
DEFAULT_FRAME_INTERVAL_PS = 10.0
REQUIRED_COLUMNS = ('system', 'metal_species', 'composition', 'frame', 'time_ps', 'metal_atom_id', 'peo_chain_eo_mapping', 'tfsi_denticity')

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Count metals sharing the same PEO chain, PEO oxygen, or TFSI anion.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--master-db', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument('--block-ns', type=float, default=DEFAULT_BLOCK_NS)
    parser.add_argument('--frame-interval-ps', type=float, default=DEFAULT_FRAME_INTERVAL_PS)
    parser.add_argument('--overwrite', action='store_true')
    return parser

def configure_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger('ligand_multimetal')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    stream = logging.StreamHandler(sys.stdout); stream.setFormatter(formatter); logger.addHandler(stream)
    file_handler = logging.FileHandler(path, mode='w', encoding='utf-8'); file_handler.setFormatter(formatter); logger.addHandler(file_handler)
    return logger

def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists():
        if overwrite: shutil.rmtree(path)
        elif any(path.iterdir()): raise FileExistsError(f'Output directory already contains files: {path}\nUse --overwrite to replace it.')
    path.mkdir(parents=True, exist_ok=True)
    return path

def parse_json_dictionary(value: Any) -> Dict[str, Any]:
    if value is None: return {}
    if isinstance(value, float) and np.isnan(value): return {}
    if isinstance(value, dict): return value
    text = str(value).strip()
    if not text or text.lower() in {'nan', 'none', 'null', '{}'}: return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict): raise ValueError(f'Expected JSON dictionary, found {type(parsed).__name__}')
    return parsed

def database_candidates(root: Path) -> List[Path]:
    patterns = ('coordination_database/*/master_coordination_*.csv.gz', 'coordination_database/master_coordination_*.csv.gz', '**/coordination_database/*/master_coordination_*.csv.gz', '**/coordination_database/master_coordination_*.csv.gz')
    found: Set[Path] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file(): found.add(path.resolve())
    return sorted(found)

def resolve_master_database(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.exists(): raise FileNotFoundError(f'Master database not found: {path}')
        return path
    candidates = database_candidates(Path.cwd())
    if len(candidates) == 1: return candidates[0]
    if not candidates: raise FileNotFoundError('No master_coordination_*.csv.gz database found. Supply the path using --master-db.')
    details = '\n'.join((f'  - {path}' for path in candidates))
    raise RuntimeError('More than one master database was found. Use --master-db.\n' + details)

def safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan

class Accumulator:
    def __init__(self, block_ns: float) -> None:
        self.block_ns = block_ns
        self.peo_chain_multiplicity = Counter(); self.peo_oxygen_multiplicity = Counter(); self.tfsi_multiplicity = Counter()
        self.block_peo_chain = defaultdict(Counter); self.block_peo_oxygen = defaultdict(Counter); self.block_tfsi = defaultdict(Counter)
        self.n_metal_frames = 0; self.n_metal_frames_with_peo = 0; self.n_metal_frames_with_tfsi = 0
        self.n_metal_frames_with_shared_peo_chain = 0; self.n_metal_frames_with_shared_peo_oxygen = 0; self.n_metal_frames_with_shared_tfsi = 0
        self.peo_chain_contacts_total = 0; self.peo_chain_contacts_shared = 0; self.peo_oxygen_contacts_total = 0
        self.peo_oxygen_contacts_shared = 0; self.tfsi_contacts_total = 0; self.tfsi_contacts_shared = 0; self.n_frames = 0

    def process_frame(self, frame_df: pd.DataFrame, time_ps: float) -> None:
        if frame_df.empty: return
        block = int(math.floor(time_ps / 1000.0 / self.block_ns))
        peo_chain_to_metals = defaultdict(set); peo_oxygen_to_metals = defaultdict(set); tfsi_to_metals = defaultdict(set)
        frame_metals = set(); metals_with_peo = set(); metals_with_tfsi = set()
        for row in frame_df.itertuples(index=False):
            metal_id = int(row.metal_atom_id); frame_metals.add(metal_id)
            peo_mapping = parse_json_dictionary(row.peo_chain_eo_mapping)
            if peo_mapping: metals_with_peo.add(metal_id)
            for chain_resid_raw, eo_indices in peo_mapping.items():
                chain_resid = int(chain_resid_raw); peo_chain_to_metals[chain_resid].add(metal_id)
                unique_eos = {int(value) for value in eo_indices} if isinstance(eo_indices, (list, tuple, set)) else set(range(int(eo_indices)))
                for eo_index in unique_eos: peo_oxygen_to_metals[chain_resid, eo_index].add(metal_id)
            tfsi_mapping = parse_json_dictionary(row.tfsi_denticity)
            if tfsi_mapping: metals_with_tfsi.add(metal_id)
            for tfsi_resid_raw in tfsi_mapping: tfsi_to_metals[int(tfsi_resid_raw)].add(metal_id)
        shared_peo_chain_metals = set(); shared_peo_oxygen_metals = set(); shared_tfsi_metals = set()
        for metal_ids in peo_chain_to_metals.values():
            multiplicity = len(metal_ids); self.peo_chain_multiplicity[multiplicity] += 1; self.block_peo_chain[block][multiplicity] += 1
            self.peo_chain_contacts_total += multiplicity
            if multiplicity >= 2: self.peo_chain_contacts_shared += multiplicity; shared_peo_chain_metals.update(metal_ids)
        for metal_ids in peo_oxygen_to_metals.values():
            multiplicity = len(metal_ids); self.peo_oxygen_multiplicity[multiplicity] += 1; self.block_peo_oxygen[block][multiplicity] += 1
            self.peo_oxygen_contacts_total += multiplicity
            if multiplicity >= 2: self.peo_oxygen_contacts_shared += multiplicity; shared_peo_oxygen_metals.update(metal_ids)
        for metal_ids in tfsi_to_metals.values():
            multiplicity = len(metal_ids); self.tfsi_multiplicity[multiplicity] += 1; self.block_tfsi[block][multiplicity] += 1
            self.tfsi_contacts_total += multiplicity
            if multiplicity >= 2: self.tfsi_contacts_shared += multiplicity; shared_tfsi_metals.update(metal_ids)
        self.n_frames += 1; self.n_metal_frames += len(frame_metals)
        self.n_metal_frames_with_peo += len(metals_with_peo); self.n_metal_frames_with_tfsi += len(metals_with_tfsi)
        self.n_metal_frames_with_shared_peo_chain += len(shared_peo_chain_metals); self.n_metal_frames_with_shared_peo_oxygen += len(shared_peo_oxygen_metals)
        self.n_metal_frames_with_shared_tfsi += len(shared_tfsi_metals)

def build_distribution(identity, ligand_type, counter):
    system, metal, composition = identity; total = sum(counter.values()); rows = []
    for multiplicity in sorted(counter):
        count = int(counter[multiplicity])
        rows.append({'system': system, 'metal_species': metal, 'composition': composition, 'ligand_type': ligand_type, 'n_coordinating_metals': int(multiplicity), 'ligand_frame_count': count, 'probability_given_coordinated_ligand': safe_fraction(count, total)})
    return pd.DataFrame(rows)

def summarize_counter(identity, ligand_type, counter, total_contacts, shared_contacts):
    system, metal, composition = identity
    total_ligand_frames = int(sum(counter.values())); shared_ligand_frames = int(sum(count for multiplicity, count in counter.items() if multiplicity >= 2))
    weighted_metals = int(sum(multiplicity * count for multiplicity, count in counter.items()))
    return {'system': system, 'metal_species': metal, 'composition': composition, 'ligand_type': ligand_type, 'n_coordinated_ligand_frames': total_ligand_frames, 'n_single_metal_ligand_frames': int(counter.get(1, 0)), 'n_multimetal_ligand_frames': shared_ligand_frames, 'fraction_single_metal_given_coordinated_ligand': safe_fraction(counter.get(1, 0), total_ligand_frames), 'fraction_multimetal_given_coordinated_ligand': safe_fraction(shared_ligand_frames, total_ligand_frames), 'mean_metals_per_coordinated_ligand': safe_fraction(weighted_metals, total_ligand_frames), 'maximum_observed_metal_multiplicity': int(max(counter)) if counter else 0, 'n_metal_ligand_contacts': int(total_contacts), 'n_contacts_belonging_to_shared_ligands': int(shared_contacts), 'fraction_contacts_belonging_to_shared_ligands': safe_fraction(shared_contacts, total_contacts)}

def build_block_distribution(identity, ligand_type, block_counts, block_ns):
    system, metal, composition = identity; rows = []
    for block in sorted(block_counts):
        counter = block_counts[block]; total = sum(counter.values()); shared = sum(count for multiplicity, count in counter.items() if multiplicity >= 2)
        weighted = sum(multiplicity * count for multiplicity, count in counter.items())
        rows.append({'system': system, 'metal_species': metal, 'composition': composition, 'ligand_type': ligand_type, 'block_index': block, 'block_start_ns': block * block_ns, 'block_end_ns': (block + 1) * block_ns, 'n_coordinated_ligand_frames': int(total), 'fraction_multimetal_given_coordinated_ligand': safe_fraction(shared, total), 'mean_metals_per_coordinated_ligand': safe_fraction(weighted, total)})
    return pd.DataFrame(rows)

def main() -> None:
    args = build_parser().parse_args()
    if args.chunk_size <= 0: raise ValueError('--chunk-size must be positive')
    if args.block_ns <= 0: raise ValueError('--block-ns must be positive')
    if args.frame_interval_ps <= 0: raise ValueError('--frame-interval-ps must be positive')
    database = resolve_master_database(args.master_db); output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    logger = configure_logger(output_dir / '08_ligand_multimetal.log')
    header = pd.read_csv(database, compression='infer', nrows=0)
    missing = [column for column in REQUIRED_COLUMNS if column not in header.columns]
    if missing: raise ValueError(f'Missing required columns: {missing}')
    accumulator = Accumulator(args.block_ns); identity = None; carry = pd.DataFrame(); rows_read = 0
    reader = pd.read_csv(database, compression='infer', usecols=list(REQUIRED_COLUMNS), chunksize=args.chunk_size, low_memory=False)
    for chunk_index, chunk in enumerate(reader, start=1):
        rows_read += len(chunk)
        if identity is None and not chunk.empty: identity = (str(chunk.iloc[0]['system']), str(chunk.iloc[0]['metal_species']), str(chunk.iloc[0]['composition']))
        chunk['frame'] = pd.to_numeric(chunk['frame'], errors='coerce'); chunk['time_ps'] = pd.to_numeric(chunk['time_ps'], errors='coerce')
        chunk['metal_atom_id'] = pd.to_numeric(chunk['metal_atom_id'], errors='raise').astype(np.int64)
        invalid_time = ~np.isfinite(chunk['time_ps'].to_numpy(float))
        if np.any(invalid_time): chunk.loc[invalid_time, 'time_ps'] = chunk.loc[invalid_time, 'frame'] * args.frame_interval_ps
        if not carry.empty: chunk = pd.concat([carry, chunk], ignore_index=True); carry = pd.DataFrame()
        last_frame = chunk['frame'].iloc[-1]; carry = chunk[chunk['frame'] == last_frame].copy(); complete = chunk[chunk['frame'] != last_frame]
        for _, frame_df in complete.groupby('frame', sort=False): accumulator.process_frame(frame_df, float(frame_df['time_ps'].iloc[0]))
    if not carry.empty: accumulator.process_frame(carry, float(carry['time_ps'].iloc[0]))
    if identity is None: raise RuntimeError('The master database contains no rows.')
    distributions = pd.concat([build_distribution(identity, 'PEO_chain', accumulator.peo_chain_multiplicity), build_distribution(identity, 'PEO_oxygen', accumulator.peo_oxygen_multiplicity), build_distribution(identity, 'TFSI_anion', accumulator.tfsi_multiplicity)], ignore_index=True)
    summaries = pd.DataFrame([summarize_counter(identity, 'PEO_chain', accumulator.peo_chain_multiplicity, accumulator.peo_chain_contacts_total, accumulator.peo_chain_contacts_shared), summarize_counter(identity, 'PEO_oxygen', accumulator.peo_oxygen_multiplicity, accumulator.peo_oxygen_contacts_total, accumulator.peo_oxygen_contacts_shared), summarize_counter(identity, 'TFSI_anion', accumulator.tfsi_multiplicity, accumulator.tfsi_contacts_total, accumulator.tfsi_contacts_shared)])
    blocks = pd.concat([build_block_distribution(identity, 'PEO_chain', accumulator.block_peo_chain, args.block_ns), build_block_distribution(identity, 'PEO_oxygen', accumulator.block_peo_oxygen, args.block_ns), build_block_distribution(identity, 'TFSI_anion', accumulator.block_tfsi, args.block_ns)], ignore_index=True)
    system, metal, composition = identity
    metal_frame_summary = pd.DataFrame([{'system': system, 'metal_species': metal, 'composition': composition, 'n_frames': accumulator.n_frames, 'n_metal_frames': accumulator.n_metal_frames, 'fraction_metal_frames_with_peo': safe_fraction(accumulator.n_metal_frames_with_peo, accumulator.n_metal_frames), 'fraction_metal_frames_with_tfsi': safe_fraction(accumulator.n_metal_frames_with_tfsi, accumulator.n_metal_frames), 'fraction_metal_frames_in_shared_peo_chain': safe_fraction(accumulator.n_metal_frames_with_shared_peo_chain, accumulator.n_metal_frames), 'fraction_peo_bound_metal_frames_in_shared_peo_chain': safe_fraction(accumulator.n_metal_frames_with_shared_peo_chain, accumulator.n_metal_frames_with_peo), 'fraction_metal_frames_sharing_same_peo_oxygen': safe_fraction(accumulator.n_metal_frames_with_shared_peo_oxygen, accumulator.n_metal_frames), 'fraction_peo_bound_metal_frames_sharing_same_peo_oxygen': safe_fraction(accumulator.n_metal_frames_with_shared_peo_oxygen, accumulator.n_metal_frames_with_peo), 'fraction_metal_frames_in_shared_tfsi_anion': safe_fraction(accumulator.n_metal_frames_with_shared_tfsi, accumulator.n_metal_frames), 'fraction_tfsi_bound_metal_frames_in_shared_tfsi_anion': safe_fraction(accumulator.n_metal_frames_with_shared_tfsi, accumulator.n_metal_frames_with_tfsi)}])
    distributions.to_csv(output_dir / 'ligand_metal_multiplicity_distribution.csv', index=False, float_format='%.10g')
    summaries.to_csv(output_dir / 'ligand_multimetal_summary.csv', index=False, float_format='%.10g')
    blocks.to_csv(output_dir / 'ligand_multimetal_block_summary.csv', index=False, float_format='%.10g')
    metal_frame_summary.to_csv(output_dir / 'metal_frame_shared_ligand_summary.csv', index=False, float_format='%.10g')
    with (output_dir / 'validation_report.json').open('w', encoding='utf-8') as fh:
        json.dump({'status': 'PASSED', 'system': {'system': system, 'metal_species': metal, 'composition': composition}, 'input': str(database), 'n_rows_read': rows_read, 'n_frames_processed': accumulator.n_frames}, fh, indent=2); fh.write('\n')
    logger.info('Output directory: %s', output_dir)
if __name__ == '__main__':
    main()
