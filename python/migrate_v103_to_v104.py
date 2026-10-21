#!/usr/bin/python3
import argparse
import torch

description = """
Migrate an Influence model version 103 checkpoint to version 104.
Version 104 reads inputs version 103 (same input counts, extra globals in slots that were always 0 before),
and its lead output (score-value channel 2) learns the moves left until the game ends.
The weights are unchanged, so this only rewrites the stored config.
Run on a torch checkpoint.ckpt file. Replace the old one with the new one after a backup.
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-checkpoint', help='Checkpoint file path', required=True)
parser.add_argument('-output', help='Output new checkpoint to here', required=True)
args = vars(parser.parse_args())

checkpoint_path = args["checkpoint"]
output_path = args["output"]

data = torch.load(checkpoint_path,map_location="cpu")
assert data["config"]["version"] == 103, f"Expected a version 103 checkpoint, got {data['config']['version']}"
data["config"]["version"] = 104

if "running_metrics" in data and "sums" in data["running_metrics"]:
    # The lead loss is replaced by the moves-left loss
    for key in ("sums", "weights"):
        if key in data["running_metrics"]:
            data["running_metrics"][key].pop("leadloss_sum", None)

print(f"Saving to {output_path}")
torch.save(data, output_path)
