#!/usr/bin/python3
"""
Compare the outputs of a C++ backend with the python model, or two C++ backends with each other.

Positions come from `katago dumpnnparity -model M.bin.gz -config C -out f.npz`, which stores the inputs
and the postprocessed outputs of the C++ evaluator.

  python test_nn_parity.py -npz f.npz -checkpoint model.ckpt     # C++ vs python
  python test_nn_parity.py -npz f.npz -other-npz g.npz           # backend vs backend (same seed and positions)
"""
import argparse
import math
import numpy as np
import torch

from load_model import load_model


def python_outputs(checkpoint, use_swa, spatial, global_input):
    nn_len = spatial.shape[2]
    model, swa_model, _ = load_model(checkpoint, use_swa, device="cpu", pos_len=nn_len, verbose=False)
    net = swa_model if swa_model is not None else model
    net.eval()
    version = model.config["version"]
    with torch.no_grad():
        outputs = net(torch.tensor(spatial), torch.tensor(global_input))
        (
            policy_logits, value_logits, _, _, _, _, _, _,
            pred_scoremean, _, pred_lead, pred_variance_time, pred_shortterm_value_error, _, _,
        ) = model.postprocess_output(outputs)[0]
    policy_logits = policy_logits[:, 0, :].double().numpy()
    value = torch.softmax(value_logits.double(), dim=1).numpy()
    moves_left = np.maximum(0.0, pred_scoremean.double().numpy())
    micro_moves_left = np.maximum(0.0, pred_lead.double().numpy()) if 104 <= version <= 199 else np.zeros_like(moves_left)
    var_time_left = pred_variance_time.double().numpy()
    # C++ reports sqrt of the python shortterm value error for these model versions
    st_wl_error = np.sqrt(pred_shortterm_value_error.double().numpy())
    misc = np.stack([moves_left, micro_moves_left, var_time_left, st_wl_error], axis=1)
    return policy_logits, value, misc


def masked_softmax(logits, legal):
    logits = np.where(legal, logits, -np.inf)
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.where(legal, np.exp(logits), 0.0)
    return e / e.sum(axis=1, keepdims=True)


def report(name, a, b, legal=None):
    if legal is not None:
        diff = np.abs(np.where(legal, a - b, 0.0))
    else:
        diff = np.abs(a - b)
    print(f"{name:16s} max abs err {diff.max():.3e}   mean abs err {diff.mean():.3e}   (magnitude {np.abs(b).mean():.3e})")
    return diff.max()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-npz", required=True, help="Output of katago dumpnnparity")
    parser.add_argument("-checkpoint", help="Python checkpoint the model was exported from")
    parser.add_argument("-use-swa", action="store_true", help="Compare against the SWA model of the checkpoint")
    parser.add_argument("-other-npz", help="Second dumpnnparity output to compare against (same positions)")
    args = parser.parse_args()

    d = dict(np.load(args.npz))
    spatial, global_input = d["binaryInputNCHW"], d["globalInputNC"]
    policy, value, misc = d["policyProbs"].astype(np.float64), d["value"].astype(np.float64), d["misc"].astype(np.float64)
    legal = policy >= 0
    print(f"{spatial.shape[0]} positions, {legal.sum(axis=1).mean():.1f} legal moves on average")

    if args.checkpoint is not None:
        py_logits, py_value, py_misc = python_outputs(args.checkpoint, args.use_swa, spatial, global_input)
        py_policy = masked_softmax(py_logits, legal)
        ref = ("python", py_policy, py_value, py_misc)
    elif args.other_npz is not None:
        o = dict(np.load(args.other_npz))
        assert np.array_equal(o["binaryInputNCHW"], spatial) and np.array_equal(o["globalInputNC"], global_input), \
            "The two files have different positions, use the same -seed and -num-positions"
        ref = ("other npz", o["policyProbs"].astype(np.float64), o["value"].astype(np.float64), o["misc"].astype(np.float64))
    else:
        raise SystemExit("Give -checkpoint or -other-npz")

    ref_name, ref_policy, ref_value, ref_misc = ref
    print(f"Comparing {args.npz} against {ref_name}")
    report("policy", policy, ref_policy, legal)
    report("value", value, ref_value)
    for i, name in enumerate(["movesLeft", "microMovesLeft", "varTimeLeft", "stWinlossError"]):
        report(name, misc[:, i], ref_misc[:, i])
    top_match = np.mean(np.argmax(np.where(legal, policy, -1), axis=1) == np.argmax(np.where(legal, ref_policy, -1), axis=1))
    print(f"top policy move agrees on {100.0 * top_match:.1f}% of positions")


if __name__ == "__main__":
    main()
