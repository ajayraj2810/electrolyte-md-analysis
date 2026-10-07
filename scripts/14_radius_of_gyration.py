#!/usr/bin/env python3

from __future__ import annotations
import argparse
from pathlib import Path
import MDAnalysis as mda
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parser():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--topology", type=Path, required=True)
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--polymer-selection", default="resid 1:40")
    p.add_argument("--chain-attribute", choices=["resid","segid"], default="resid")
    p.add_argument("--atom-slice-start", type=int, default=0,
                   help="Optional first atom within each chain used for Rg; 0 uses the full chain")
    p.add_argument("--bins", type=int, default=40)
    p.add_argument("--output-dir", type=Path, default=Path("14_radius_of_gyration"))
    return p

def main():
    args=parser().parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    u=mda.Universe(str(args.topology),str(args.trajectory),format="LAMMPSDUMP")
    group=u.select_atoms(args.polymer_selection)
    labels=np.unique(group.resids if args.chain_attribute=="resid" else group.segids)
    rows=[]
    for frame,ts in enumerate(u.trajectory):
        values=[]
        for label in labels:
            chain=group[group.resids==label] if args.chain_attribute=="resid" else group[group.segids==label]
            if args.atom_slice_start: chain=chain[args.atom_slice_start:]
            if len(chain)==0: continue
            rg=float(chain.radius_of_gyration(wrap=False)); values.append(rg)
            rows.append({"frame":frame,"chain":str(label),"Rg_A":rg})
    df=pd.DataFrame(rows); df.to_csv(args.output_dir/"radius_of_gyration_by_chain.csv",index=False)
    if df.empty: raise RuntimeError("No polymer-chain Rg values were generated; check the selection.")
    frame_mean=df.groupby("frame",as_index=False)["Rg_A"].mean().rename(columns={"Rg_A":"mean_Rg_A"})
    frame_mean.to_csv(args.output_dir/"radius_of_gyration_frame_mean.csv",index=False)
    plt.figure(figsize=(6,4.5)); plt.hist(df["Rg_A"],bins=args.bins,density=True)
    plt.xlabel("Radius of gyration (Å)"); plt.ylabel("Probability density")
    plt.tight_layout(); plt.savefig(args.output_dir/"radius_of_gyration_distribution.png",dpi=300); plt.close()
    print(f"Mean chain Rg = {df['Rg_A'].mean():.4f} Å")

if __name__=="__main__": main()