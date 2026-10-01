#!/usr/bin/env python3
import argparse, os, subprocess, sys
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument('--pair', required=True)
ap.add_argument('--pairs', type=Path, required=True)
ap.add_argument('--num-shards', type=int, required=True)
ap.add_argument('--shard-index', type=int, required=True)
ap.add_argument('--output-dir', type=Path, required=True)
ap.add_argument('--train-gold', type=Path, required=True)
ap.add_argument('--data-root', type=Path, required=True)
a=ap.parse_args()
a.output_dir.mkdir(parents=True, exist_ok=True)
env=os.environ.copy(); env['METAMATCH_BIOML_DIR']=str(a.data_root)
subprocess.run([sys.executable,'-m','src.scripts.union403_2026','shard-csv',
 '--pairs',str(a.pairs),'--num-shards',str(a.num_shards),'--shard-index',str(a.shard_index),
 '--output',str(a.output_dir/'pairs.csv')], check=True, env=env)
subprocess.run([sys.executable,'-m','src.scripts.analyze_oracle_lexical_structural_2025',
 '--pair',a.pair,'--candidates',str(a.output_dir/'pairs.csv'),'--train-gold',str(a.train_gold),
 '--output-dir',str(a.output_dir),'--depth','4','--max-literals','64','--max-related','64',
 '--workers','1','--features-only','--skip-missed-gold','--skip-oracle-cv'], check=True, env=env)
