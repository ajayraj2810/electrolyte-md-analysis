#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_exchange_vs_translation.py
=============================

Exchange-versus-translation analysis for Mg2+/Ca2+ in
PEO/EMIM-TFSI/M(TFSI)2 electrolytes.

Question
--------
Does a metal ion translate while retaining the same coordination environment
(vehicular-like motion), or is large displacement associated with ligand-shell
exchange/reorganization?

For each metal, time origin t0, and lag t:

    dr2(t) = |r_M(t0+t) - r_M(t0)|^2

and ligand retention:

    f_ret(t) = |N(t0) intersect N(t0+t)| / |N(t0)|

are calculated for:
    TFSI molecular identity,
    PEO-chain identity,
    PEO EO identity within chain,
    combined molecular shell = TFSI molecules + PEO chains.

The code also computes endpoint ligand loss/gain, cumulative shell turnover,
coordination-state-resolved mobility (P/PT/T/F), MSD conditioned on retention,
and the exchange-enhancement ratio:

    E_exch = MSD(f_ret < 0.5) / MSD(f_ret >= 0.8)

Inputs
------
1. Master coordination database from 01_build_master_coordination_database.py
   Required columns:
       system, metal_species, composition, frame, time_ps, metal_atom_id,
       peo_chain_eo_mapping, tfsi_denticity

2. LAMMPS topology/data file and trajectory.
   The trajectory is read ONLY for metal coordinates and periodic box data.
   Coordination is NOT reconstructed from the trajectory.

Typical execution
-----------------
python 10_exchange_vs_translation.py --overwrite

or explicitly:

python 10_exchange_vs_translation.py \
    --master-db coordination_database/<system>/master_coordination_<system>.csv.gz \
    --topology system.data \
    --trajectory system.lammpsdump \
    --overwrite

Use --write-raw only when you need every origin-lag sample; it can be large.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import MDAnalysis as mda
except ImportError as exc:
    raise ImportError(
        "MDAnalysis is required. Activate the same environment used for the "
        "other trajectory-analysis modules."
    ) from exc

DEFAULT_OUTPUT_DIR = Path("10_exchange_vs_translation")
DEFAULT_METAL_ATOM_TYPE = 17
DEFAULT_FRAME_DT_PS = 10.0
DEFAULT_BLOCK_NS = 10.0
DEFAULT_FIGURE_DPI = 300
DEFAULT_LAGS_PS = (10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0)
RETENTION_EDGES = np.array([-1e-12, 0.20, 0.40, 0.60, 0.80, 1.0000000001])
RETENTION_LABELS = ("0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0")
SHELL_TYPES = ("TFSI", "PEO_chain", "PEO_EO", "combined")
REQUIRED_MASTER_COLUMNS = (
    "system", "metal_species", "composition", "frame", "time_ps",
    "metal_atom_id", "peo_chain_eo_mapping", "tfsi_denticity",
)

@dataclass
class SystemData:
    system: str
    metal_species: str
    composition: str
    frames: np.ndarray
    times_ps: np.ndarray
    metal_ids: np.ndarray
    tfsi_sets: List[List[Set[str]]]
    peo_chain_sets: List[List[Set[str]]]
    peo_eo_sets: List[List[Set[str]]]
    state: np.ndarray


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Relate metal translation to coordination-shell retention/exchange.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--master-db", type=Path, default=None)
    p.add_argument("--topology", type=Path, default=None)
    p.add_argument("--trajectory", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--metal-atom-type", type=int, default=DEFAULT_METAL_ATOM_TYPE)
    p.add_argument("--frame-dt-ps", type=float, default=DEFAULT_FRAME_DT_PS)
    p.add_argument("--lags-ps", type=float, nargs="+", default=list(DEFAULT_LAGS_PS))
    p.add_argument("--block-ns", type=float, default=DEFAULT_BLOCK_NS)
    p.add_argument("--figure-dpi", type=int, default=DEFAULT_FIGURE_DPI)
    p.add_argument("--write-raw", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def unique_files(paths: Iterable[Path]) -> List[Path]:
    return sorted(set(p.resolve() for p in paths if p.exists() and p.is_file()), key=str)


def discover_master(explicit: Optional[Path], base: Path) -> Path:
    if explicit is not None:
        p = explicit if explicit.is_absolute() else base / explicit
        p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    roots = [base / "coordination_database", base / "01_database", base]
    candidates = []
    for root in roots:
        if not root.exists():
            continue
        for pat in ("**/master_coordination*.csv.gz", "**/master_database.csv.gz", "**/*master*coordination*.csv.gz"):
            candidates.extend(root.glob(pat))
    viable = []
    for p in unique_files(candidates):
        try:
            cols = set(pd.read_csv(p, compression="infer", nrows=0).columns)
        except Exception:
            continue
        if set(REQUIRED_MASTER_COLUMNS).issubset(cols):
            viable.append(p)
    if len(viable) == 1:
        return viable[0]
    if not viable:
        raise FileNotFoundError("No compatible master database found; use --master-db.")
    raise RuntimeError("Multiple master databases found; use --master-db:\n" + "\n".join(map(str, viable)))


def discover_topology(explicit: Optional[Path], base: Path) -> Path:
    if explicit is not None:
        p = explicit if explicit.is_absolute() else base / explicit
        p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    c = unique_files(list(base.glob("*.data")) + list(base.glob("*.lmp")) + list(base.glob("*.lammps")))
    c = [p for p in c if "dump" not in p.name.lower() and "traj" not in p.name.lower()]
    if len(c) == 1:
        return c[0]
    raise RuntimeError("Use --topology explicitly. Candidates:\n" + "\n".join(map(str, c)))


def discover_trajectory(explicit: Optional[Path], base: Path) -> Path:
    if explicit is not None:
        p = explicit if explicit.is_absolute() else base / explicit
        p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    c = []
    for pat in ("*.lammpsdump", "*.lammpstrj", "*.dump", "*.dcd", "*.xtc", "*.trr"):
        c.extend(base.glob(pat))
    c = unique_files(c)
    if len(c) == 1:
        return c[0]
    raise RuntimeError("Use --trajectory explicitly. Candidates:\n" + "\n".join(map(str, c)))


def setup_output(out: Path, overwrite: bool) -> None:
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} exists; use --overwrite.")
        shutil.rmtree(out)
    (out / "figures").mkdir(parents=True)
    (out / "raw").mkdir(parents=True)


def setup_logger(path: Path) -> logging.Logger:
    log = logging.getLogger("exchange_vs_translation")
    log.handlers.clear(); log.setLevel(logging.INFO); log.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(path, mode="w", encoding="utf-8"); fh.setFormatter(fmt)
    log.addHandler(sh); log.addHandler(fh)
    return log


def parse_mapping(value: Any) -> Dict[str, Any]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "{}"}:
        return {}
    try:
        obj = json.loads(text)
    except Exception:
        obj = ast.literal_eval(text)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected mapping, got: {text[:100]}")
    return obj


def make_peo_sets(value: Any) -> Tuple[Set[str], Set[str]]:
    mapping = parse_mapping(value)
    chains, eos = set(), set()
    for chain, vals in mapping.items():
        c = str(chain); chains.add(c)
        if not isinstance(vals, (list, tuple, set)):
            vals = [vals]
        for eo in vals:
            eos.add(f"{c}:{eo}")
    return chains, eos


def make_tfsi_set(value: Any) -> Set[str]:
    return set(str(k) for k in parse_mapping(value).keys())


def state_of(peo_chains: Set[str], tfsi: Set[str]) -> str:
    if peo_chains and tfsi: return "PT"
    if peo_chains: return "P"
    if tfsi: return "T"
    return "F"


def load_master(path: Path, log: logging.Logger) -> SystemData:
    df = pd.read_csv(path, compression="infer", usecols=list(REQUIRED_MASTER_COLUMNS), low_memory=False)
    if df.empty:
        raise ValueError("Master database is empty.")
    df["frame"] = pd.to_numeric(df["frame"], errors="raise").astype(int)
    df["time_ps"] = pd.to_numeric(df["time_ps"], errors="raise").astype(float)
    df["metal_atom_id"] = pd.to_numeric(df["metal_atom_id"], errors="raise").astype(int)
    if df.duplicated(["frame", "metal_atom_id"]).any():
        raise ValueError("Duplicate frame/metal rows in master database.")
    system = df["system"].astype(str).unique(); metal = df["metal_species"].astype(str).unique(); comp = df["composition"].astype(str).unique()
    if len(system)!=1 or len(metal)!=1 or len(comp)!=1:
        raise ValueError("Master DB must contain exactly one system/metal/composition.")
    frames = np.sort(df.frame.unique()); metals = np.sort(df.metal_atom_id.unique())
    if len(df) != len(frames)*len(metals):
        raise ValueError("Master DB is not a complete metal-by-frame grid.")
    fidx = {int(v):i for i,v in enumerate(frames)}; midx={int(v):i for i,v in enumerate(metals)}
    tfsi = [[set() for _ in metals] for _ in frames]
    pchain = [[set() for _ in metals] for _ in frames]
    peo = [[set() for _ in metals] for _ in frames]
    states = np.empty((len(frames), len(metals)), dtype="U2")
    times = (df[["frame","time_ps"]].drop_duplicates("frame").sort_values("frame").time_ps.to_numpy(float))
    for r in df.itertuples(index=False):
        fi=fidx[int(r.frame)]; mi=midx[int(r.metal_atom_id)]
        t=make_tfsi_set(r.tfsi_denticity); pc,eo=make_peo_sets(r.peo_chain_eo_mapping)
        tfsi[fi][mi]=t; pchain[fi][mi]=pc; peo[fi][mi]=eo; states[fi,mi]=state_of(pc,t)
    log.info("Master DB: %d frames, %d metals", len(frames), len(metals))
    return SystemData(str(system[0]),str(metal[0]),str(comp[0]),frames,times,metals,tfsi,pchain,peo,states)


def box_matrix(dim: np.ndarray) -> np.ndarray:
    a,b,c,ad,bd,gd = map(float, dim); al=np.deg2rad(ad); be=np.deg2rad(bd); ga=np.deg2rad(gd)
    sg=np.sin(ga)
    av=np.array([a,0,0.]); bv=np.array([b*np.cos(ga), b*sg,0.])
    cx=c*np.cos(be); cy=c*(np.cos(al)-np.cos(be)*np.cos(ga))/sg; cz=np.sqrt(max(c*c-cx*cx-cy*cy,0.))
    return np.vstack([av,bv,np.array([cx,cy,cz])])


def read_unwrapped_positions(topology: Path, trajectory: Path, data: SystemData, metal_type: int, log: logging.Logger):
    u = mda.Universe(str(topology), str(trajectory))
    ag = u.select_atoms(f"type {metal_type}")
    if len(ag)==0: raise ValueError(f"No atoms selected with type {metal_type}.")
    ids=np.asarray(ag.ids,int); lookup={int(v):i for i,v in enumerate(ids)}
    missing=[int(v) for v in data.metal_ids if int(v) not in lookup]
    if missing: raise ValueError(f"Master metal IDs missing from trajectory: {missing[:20]}")
    order=np.array([lookup[int(v)] for v in data.metal_ids],int)
    if data.frames.min()<0 or data.frames.max()>=len(u.trajectory):
        raise IndexError("Master frame numbers do not map directly to trajectory frame indices.")
    wrapped=np.empty((len(data.frames),len(data.metal_ids),3),float); boxes=np.empty((len(data.frames),6),float)
    for i,f in enumerate(data.frames):
        ts=u.trajectory[int(f)]; wrapped[i]=ag.positions[order]; boxes[i]=ts.dimensions[:6]
        if i%1000==0 or i==len(data.frames)-1: log.info("Read coordinates %d/%d",i+1,len(data.frames))
    un=np.empty_like(wrapped); un[0]=wrapped[0]; maxstep=0.0
    for i in range(1,len(data.frames)):
        avg=0.5*(boxes[i-1]+boxes[i]); d=wrapped[i]-wrapped[i-1]
        if np.allclose(avg[3:6],90.,atol=1e-4,rtol=0):
            L=avg[:3]; dm=d-L*np.rint(d/L)
        else:
            H=box_matrix(avg); inv=np.linalg.inv(H); frac=d@inv; frac-=np.rint(frac); dm=frac@H
        un[i]=un[i-1]+dm; maxstep=max(maxstep,float(np.linalg.norm(dm,axis=1).max()))
    return un, maxstep


def retention(start: Set[str], end: Set[str]) -> Dict[str,float]:
    n0=len(start); n1=len(end); nr=len(start&end); lost=len(start-end); gained=len(end-start); union=len(start|end)
    return {"n_initial":n0,"n_final":n1,"n_retained":nr,"n_lost":lost,"n_gained":gained,
            "endpoint_exchange_count":lost+gained,
            "retention_fraction":nr/n0 if n0 else np.nan,
            "jaccard":nr/union if union else np.nan}


def combined(tfsi:Set[str], pchain:Set[str])->Set[str]:
    return {f"T:{x}" for x in tfsi}|{f"P:{x}" for x in pchain}


def cumulative_turnover(grid: List[List[Set[str]]], fi0:int, fi1:int, mi:int)->int:
    total=0; prev=grid[fi0][mi]
    for fi in range(fi0+1,fi1+1):
        cur=grid[fi][mi]; total+=len(prev^cur); prev=cur
    return total


def cumulative_combined(data:SystemData,fi0:int,fi1:int,mi:int)->int:
    total=0; prev=combined(data.tfsi_sets[fi0][mi],data.peo_chain_sets[fi0][mi])
    for fi in range(fi0+1,fi1+1):
        cur=combined(data.tfsi_sets[fi][mi],data.peo_chain_sets[fi][mi]); total+=len(prev^cur); prev=cur
    return total


def ret_bin(x:float):
    if not np.isfinite(x): return None
    i=np.searchsorted(RETENTION_EDGES,x,side="right")-1; i=min(max(i,0),len(RETENTION_LABELS)-1)
    return RETENTION_LABELS[i]


def frame_dt(data:SystemData,fallback:float)->float:
    d=np.diff(data.times_ps); d=d[np.isfinite(d)&(d>0)]
    return float(np.median(d)) if len(d) else float(fallback)


def lag_frames(lags_ps: Sequence[float], dt:float,nframes:int)->List[int]:
    vals=sorted(set(int(round(x/dt)) for x in lags_ps if int(round(x/dt))>=1 and int(round(x/dt))<nframes))
    if not vals: raise ValueError("No valid requested lag.")
    return vals


def calculate_samples(data:SystemData,pos:np.ndarray,lags:Sequence[int],dt:float,block_ns:float,log:logging.Logger)->pd.DataFrame:
    rec=[]; block_ps=block_ns*1000.; nF,nM,_=pos.shape
    for lf in lags:
        log.info("Lag %.4g ns",lf*dt/1000.)
        for fi0 in range(nF-lf):
            fi1=fi0+lf; disp=pos[fi1]-pos[fi0]; dr2=np.einsum("ij,ij->i",disp,disp); dr=np.sqrt(dr2)
            block=int(math.floor(data.times_ps[fi0]/block_ps)) if block_ps>0 else 0
            for mi in range(nM):
                shell_pairs={
                    "TFSI":(data.tfsi_sets[fi0][mi],data.tfsi_sets[fi1][mi]),
                    "PEO_chain":(data.peo_chain_sets[fi0][mi],data.peo_chain_sets[fi1][mi]),
                    "PEO_EO":(data.peo_eo_sets[fi0][mi],data.peo_eo_sets[fi1][mi]),
                    "combined":(combined(data.tfsi_sets[fi0][mi],data.peo_chain_sets[fi0][mi]),combined(data.tfsi_sets[fi1][mi],data.peo_chain_sets[fi1][mi])),
                }
                for shell,(s0,s1) in shell_pairs.items():
                    rm=retention(s0,s1)
                    if shell=="TFSI": turn=cumulative_turnover(data.tfsi_sets,fi0,fi1,mi)
                    elif shell=="PEO_chain": turn=cumulative_turnover(data.peo_chain_sets,fi0,fi1,mi)
                    elif shell=="PEO_EO": turn=cumulative_turnover(data.peo_eo_sets,fi0,fi1,mi)
                    else: turn=cumulative_combined(data,fi0,fi1,mi)
                    rec.append({"system":data.system,"metal_species":data.metal_species,"composition":data.composition,
                                "metal_id":int(data.metal_ids[mi]),"origin_frame":int(data.frames[fi0]),"origin_time_ps":float(data.times_ps[fi0]),
                                "lag_frames":lf,"lag_ps":lf*dt,"lag_ns":lf*dt/1000.,"origin_block":block,
                                "origin_state":str(data.state[fi0,mi]),"final_state":str(data.state[fi1,mi]),"shell_type":shell,
                                "displacement_A":float(dr[mi]),"displacement_sq_A2":float(dr2[mi]),**rm,
                                "retention_bin":ret_bin(rm["retention_fraction"]),"cumulative_turnover_count":int(turn)})
    return pd.DataFrame(rec)


def summarize(samples:pd.DataFrame):
    valid=samples[np.isfinite(samples.retention_fraction)].copy()
    byret=(valid.groupby(["system","metal_species","composition","shell_type","lag_ps","lag_ns","retention_bin"],observed=True)
           .agg(n_samples=("displacement_sq_A2","size"),n_metals=("metal_id","nunique"),n_origin_blocks=("origin_block","nunique"),
                mean_retention_fraction=("retention_fraction","mean"),mean_displacement_sq_A2=("displacement_sq_A2","mean"),
                mean_displacement_A=("displacement_A","mean"),mean_endpoint_exchange_count=("endpoint_exchange_count","mean"),
                mean_cumulative_turnover_count=("cumulative_turnover_count","mean")).reset_index())
    blockret=(valid.groupby(["shell_type","lag_ps","retention_bin","origin_block"],observed=True).displacement_sq_A2.mean().reset_index())
    sem=(blockret.groupby(["shell_type","lag_ps","retention_bin"],observed=True).displacement_sq_A2.agg(["std","count"]).reset_index())
    sem["block_sem_A2"]=sem["std"]/np.sqrt(sem["count"])
    byret=byret.merge(sem[["shell_type","lag_ps","retention_bin","block_sem_A2"]],on=["shell_type","lag_ps","retention_bin"],how="left")
    bystate=(samples.groupby(["system","metal_species","composition","shell_type","lag_ps","lag_ns","origin_state"],observed=True)
             .agg(n_samples=("displacement_sq_A2","size"),mean_displacement_sq_A2=("displacement_sq_A2","mean"),
                  mean_displacement_A=("displacement_A","mean"),mean_retention_fraction=("retention_fraction","mean"),
                  mean_endpoint_exchange_count=("endpoint_exchange_count","mean"),mean_cumulative_turnover_count=("cumulative_turnover_count","mean")).reset_index())
    bylag=(samples.groupby(["system","metal_species","composition","shell_type","lag_ps","lag_ns"],observed=True)
           .agg(n_samples=("displacement_sq_A2","size"),mean_displacement_sq_A2=("displacement_sq_A2","mean"),
                mean_displacement_A=("displacement_A","mean"),mean_retention_fraction=("retention_fraction","mean"),
                median_retention_fraction=("retention_fraction","median"),mean_endpoint_exchange_count=("endpoint_exchange_count","mean"),
                mean_cumulative_turnover_count=("cumulative_turnover_count","mean")).reset_index())
    erows=[]
    for keys,g in valid.groupby(["system","metal_species","composition","shell_type","lag_ps","lag_ns"],observed=True):
        lo=g[g.retention_fraction<0.5]; hi=g[g.retention_fraction>=0.8]
        ml=lo.displacement_sq_A2.mean() if len(lo) else np.nan; mh=hi.displacement_sq_A2.mean() if len(hi) else np.nan
        erows.append({"system":keys[0],"metal_species":keys[1],"composition":keys[2],"shell_type":keys[3],"lag_ps":keys[4],"lag_ns":keys[5],
                      "n_low_retention":len(lo),"n_high_retention":len(hi),"msd_low_retention_A2":ml,"msd_high_retention_A2":mh,
                      "exchange_enhancement":ml/mh if np.isfinite(ml) and np.isfinite(mh) and mh>0 else np.nan})
    enh=pd.DataFrame(erows)
    blocks=(samples.groupby(["system","metal_species","composition","shell_type","lag_ps","lag_ns","origin_block","origin_state"],observed=True)
            .agg(n_samples=("displacement_sq_A2","size"),mean_displacement_sq_A2=("displacement_sq_A2","mean"),
                 mean_retention_fraction=("retention_fraction","mean"),mean_cumulative_turnover_count=("cumulative_turnover_count","mean")).reset_index())
    return byret,bystate,bylag,enh,blocks


def plots(byret,bystate,bylag,enh,samples,out:Path,dpi:int):
    for shell in ("TFSI","PEO_chain"):
        d=byret[byret.shell_type==shell]; avail=np.sort(d.lag_ns.unique()); targets=[]
        for t in (0.1,1.,10.):
            if len(avail): targets.append(float(avail[np.argmin(abs(avail-t))]))
        fig,ax=plt.subplots(figsize=(7.4,5.1)); x=np.arange(len(RETENTION_LABELS))
        for lag in sorted(set(targets)):
            s=d[np.isclose(d.lag_ns,lag)].set_index("retention_bin").reindex(RETENTION_LABELS)
            y=s.mean_displacement_sq_A2.to_numpy(float); e=s.block_sem_A2.to_numpy(float); m=np.isfinite(y)
            ax.errorbar(x[m],y[m],yerr=e[m] if np.isfinite(e[m]).any() else None,marker="o",capsize=2,label=f"{lag:g} ns")
        ax.set_xticks(x);ax.set_xticklabels(RETENTION_LABELS);ax.set_xlabel("Fraction of original shell retained");ax.set_ylabel(r"Metal MSD ($\AA^2$)")
        ax.set_title(f"Metal translation conditioned on {shell} retention");ax.legend(frameon=False);fig.tight_layout()
        fig.savefig(out/f"retention_vs_msd_{shell}.png",dpi=dpi);fig.savefig(out/f"retention_vs_msd_{shell}.pdf");plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.4,5.1))
    for shell in SHELL_TYPES:
        s=bylag[bylag.shell_type==shell].sort_values("lag_ns");ax.plot(s.lag_ns,s.mean_retention_fraction,marker="o",label=shell)
    ax.set_xscale("log");ax.set_ylim(0,1.02);ax.set_xlabel("Lag time (ns)");ax.set_ylabel("Mean fraction of original shell retained");ax.set_title("Coordination-shell memory");ax.legend(frameon=False);fig.tight_layout()
    fig.savefig(out/"retention_memory_vs_lag.png",dpi=dpi);fig.savefig(out/"retention_memory_vs_lag.pdf");plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.4,5.1))
    for shell in ("TFSI","PEO_chain","combined"):
        s=enh[enh.shell_type==shell].sort_values("lag_ns");ax.plot(s.lag_ns,s.exchange_enhancement,marker="o",label=shell)
    ax.axhline(1.,linewidth=1);ax.set_xscale("log");ax.set_xlabel("Lag time (ns)");ax.set_ylabel(r"$E_{exch}=MSD(f_{ret}<0.5)/MSD(f_{ret}\geq0.8)$");ax.set_title("Displacement enhancement associated with shell exchange");ax.legend(frameon=False);fig.tight_layout()
    fig.savefig(out/"exchange_enhancement_vs_lag.png",dpi=dpi);fig.savefig(out/"exchange_enhancement_vs_lag.pdf");plt.close(fig)
    d=samples[samples.shell_type=="combined"].copy(); avail=np.sort(d.lag_ns.unique()); lag=float(avail[np.argmin(abs(avail-1.0))]);d=d[np.isclose(d.lag_ns,lag)]
    q=np.unique(np.quantile(d.cumulative_turnover_count,[0,.2,.4,.6,.8,1.]));
    if len(q)>=3:
        d["turn_bin"]=pd.cut(d.cumulative_turnover_count,q,include_lowest=True,duplicates="drop");s=d.groupby("turn_bin",observed=True).agg(mean_turn=("cumulative_turnover_count","mean"),mean_msd=("displacement_sq_A2","mean")).reset_index()
        fig,ax=plt.subplots(figsize=(6.8,4.8));ax.plot(s.mean_turn,s.mean_msd,marker="o");ax.set_xlabel("Mean cumulative shell-turnover count");ax.set_ylabel(r"Metal MSD ($\AA^2$)");ax.set_title(f"Metal motion versus shell turnover ({lag:g} ns)");fig.tight_layout();fig.savefig(out/"displacement_vs_cumulative_turnover.png",dpi=dpi);fig.savefig(out/"displacement_vs_cumulative_turnover.pdf");plt.close(fig)
    d=bystate[bystate.shell_type=="combined"];avail=np.sort(d.lag_ns.unique());lag=float(avail[np.argmin(abs(avail-1.0))]);s=d[np.isclose(d.lag_ns,lag)].set_index("origin_state").reindex(["P","PT","T","F"])
    fig,ax=plt.subplots(figsize=(6.3,4.7));ax.bar(np.arange(4),s.mean_displacement_sq_A2.to_numpy(float));ax.set_xticks(np.arange(4));ax.set_xticklabels(["P","PT","T","F"]);ax.set_ylabel(r"Metal MSD ($\AA^2$)");ax.set_title(f"Origin-state-conditioned metal mobility ({lag:g} ns)");fig.tight_layout();fig.savefig(out/"state_resolved_msd.png",dpi=dpi);fig.savefig(out/"state_resolved_msd.pdf");plt.close(fig)


def main():
    args=parser().parse_args();base=Path.cwd();master=discover_master(args.master_db,base);top=discover_topology(args.topology,base);traj=discover_trajectory(args.trajectory,base)
    out=args.output_dir if args.output_dir.is_absolute() else base/args.output_dir;out=out.resolve();setup_output(out,args.overwrite);log=setup_logger(out/"10_exchange_vs_translation.log");t0=time.perf_counter()
    log.info("MODULE 10: EXCHANGE VERSUS TRANSLATION");log.info("Master: %s",master);log.info("Topology: %s",top);log.info("Trajectory: %s",traj)
    data=load_master(master,log);dt=frame_dt(data,args.frame_dt_ps);lags=lag_frames(args.lags_ps,dt,len(data.frames));log.info("Frame dt %.4f ps | lags %s",dt,[x*dt/1000 for x in lags])
    pos,maxstep=read_unwrapped_positions(top,traj,data,args.metal_atom_type,log);samples=calculate_samples(data,pos,lags,dt,args.block_ns,log)
    byret,bystate,bylag,enh,blocks=summarize(samples)
    byret.to_csv(out/"exchange_translation_by_retention.csv",index=False);bystate.to_csv(out/"exchange_translation_by_state.csv",index=False);bylag.to_csv(out/"exchange_translation_by_lag.csv",index=False);enh.to_csv(out/"exchange_translation_exchange_enhancement.csv",index=False);blocks.to_csv(out/"exchange_translation_block_summary.csv",index=False)
    bylag[["system","metal_species","composition","shell_type","lag_ps","lag_ns","mean_retention_fraction","median_retention_fraction"]].to_csv(out/"exchange_translation_retention_memory.csv",index=False)
    if args.write_raw:samples.to_csv(out/"raw"/"exchange_translation_samples.csv.gz",index=False,compression="gzip")
    plots(byret,bystate,bylag,enh,samples,out/"figures",args.figure_dpi)
    checks={"sample_table_nonempty":len(samples)>0,"displacements_nonnegative":bool((samples.displacement_sq_A2>=-1e-12).all()),"retention_bounds":bool(samples.retention_fraction.dropna().between(-1e-12,1+1e-12).all()),"jaccard_bounds":bool(samples.jaccard.dropna().between(-1e-12,1+1e-12).all())}
    warnings=[]
    if maxstep>25:warnings.append(f"Maximum frame-to-frame metal step {maxstep:.3f} A exceeds 25 A; inspect PBC/trajectory mapping.")
    status="FAILED" if not all(checks.values()) else ("PASSED_WITH_WARNINGS" if warnings else "PASSED")
    report={"status":status,"checks":checks,"metrics":{"system":data.system,"metal_species":data.metal_species,"composition":data.composition,"n_frames":len(data.frames),"n_metals":len(data.metal_ids),"frame_dt_ps":dt,"lags_ps":[x*dt for x in lags],"n_sample_rows":len(samples),"maximum_metal_frame_step_A":maxstep,"raw_written":bool(args.write_raw)},"warnings":warnings}
    json.dump(report,open(out/"validation_report.json","w"),indent=2)
    summary={"module":"10_exchange_vs_translation","system":{"system":data.system,"metal_species":data.metal_species,"composition":data.composition},"definitions":{"retention_fraction":"|N(t0) intersection N(t0+t)| / |N(t0)|","endpoint_exchange_count":"lost + gained between interval endpoints","cumulative_turnover_count":"sum of consecutive-frame symmetric differences","exchange_enhancement":"MSD(f_ret<0.5)/MSD(f_ret>=0.8)"},"runtime_seconds":time.perf_counter()-t0,"validation_status":status}
    json.dump(summary,open(out/"summary.json","w"),indent=2)
    log.info("DONE | validation=%s | output=%s | runtime=%.2f min",status,out,(time.perf_counter()-t0)/60)
    if status=="FAILED":raise RuntimeError("Validation failed; inspect validation_report.json")

if __name__=="__main__":
    main()