
import os
import re
import datetime

def parse_video(basename):
    match = re.search(r'PXL_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})(\d{3})', basename)
    if match:
        y, m, d, H, M, S, ms = map(int, match.groups())
        dt = datetime.datetime(y, m, d, H, M, S, ms*1000, tzinfo=datetime.timezone.utc)
        return dt.timestamp()
    return 0

def parse_log(basename):
    match = re.search(r'_(\d{8})T(\d{6})', basename)
    if match:
        date_part, time_part = match.groups()
        y, m, d = int(date_part[:4]), int(date_part[4:6]), int(date_part[6:])
        H, M, S = int(time_part[:2]), int(time_part[2:4]), int(time_part[4:])
        dt = datetime.datetime(y, m, d, H, M, S)
        return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    return 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    
    tv = parse_video(os.path.basename(args.video))
    tl = parse_log(os.path.basename(args.log))
    
    # Timezone adjust
    diff = tl - tv
    offset_h = round(diff / 3600.0)
    tl_adj = tl - offset_h * 3600.0
    
    print(f"Video: {tv:.3f}")
    print(f"Log:   {tl:.3f} (Raw)")
    print(f"Diff:  {diff:.3f}")
    print(f"LogAdj:{tl_adj:.3f}")
    print(f"Delta: {tv - tl_adj:.3f}s (Video starts THIS much later than Log)")
