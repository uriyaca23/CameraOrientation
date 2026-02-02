
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader import DataLoader
import argparse

def check(log, video):
    loader = DataLoader()
    # Load with 0 offset just to see prints
    try:
        loader.load_data(log, video, additional_offset_s=0.0)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    check(args.log, args.video)
