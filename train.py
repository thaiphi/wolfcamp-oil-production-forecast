"""Train, tune, evaluate and save the complete pipeline in a separate run folder."""
import argparse
import json
from pathlib import Path
from src.wolfcamp.training import run
from src.wolfcamp.data import DEFAULT_DATA

ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/reproduction')
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    args=parser.parse_args()
    print(json.dumps(run(args.output,args.data),indent=2))
