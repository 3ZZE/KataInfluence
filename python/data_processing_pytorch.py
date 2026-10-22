import logging
import os

import numpy as np

import torch
import torch.nn.functional

import modelconfigs

# Influence inputs version 103 globals added on top of version 102, see the layout above
# NNInputs::fillRowV102 in cpp/neuralnet/nninputs.cpp. Keep in sync with nninputs.h.
V103_POINTS_LEFT_START = 14
V103_POINTS_LEFT_UPPER_BOUNDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 24, 32, 64])  # last bucket: above 64
V103_VALUE_RATIO = 30
V103_CAN_ATTACK = 31
V103_MAX_VALUE_RATIO = 32.0
RULE_FEATURES_START = 32
NUM_RULE_FEATURES = 16
SMALL_CAP = 8

def get_row_inputs_versions(globalTargetsNC):
    """Inputs version of each row, from the C47 marker written by C++ (0 in older data, which is version 102)."""
    marker = globalTargetsNC[:, 47]
    versions = np.where(marker == 0, 102, marker).astype(np.int64)
    known = (versions == 102) | (versions == 103)
    if not np.all(known):
        raise ValueError(f"Unknown inputs version marker(s) in training data: {np.unique(marker[~known])}")
    return versions

def convert_inputs_for_model(binaryInputNCHW, globalInputNC, globalTargetsNC, model_inputs_version):
    """Converts global inputs in place, row by row, to the inputs version the model was built for.
    V102 -> V103 is exact: the new globals are functions of the V102 inputs (old selfplay never used rule options).
    V103 -> V102 drops them."""
    row_versions = get_row_inputs_versions(globalTargetsNC)
    if model_inputs_version == 102:
        rows = row_versions == 103
        globalInputNC[rows, V103_POINTS_LEFT_START:] = 0.0
    elif model_inputs_version == 103:
        rows = row_versions == 102
        if np.any(rows):
            g = globalInputNC[rows]
            g[:, V103_POINTS_LEFT_START:] = 0.0
            is_reinforce = g[:, 2] == 1.0
            points_left = np.rint(g[:, 3] * 32.0).astype(np.int64)
            reinforce_rows = np.nonzero(is_reinforce & (points_left >= 1))[0]
            buckets = np.searchsorted(V103_POINTS_LEFT_UPPER_BOUNDS, points_left[reinforce_rows], side="left")
            g[reinforce_rows, V103_POINTS_LEFT_START + buckets] = 1.0

            num_cells = np.maximum(1, np.rint(binaryInputNCHW[rows, 0].sum(axis=(1, 2))).astype(np.int64))
            own_value = np.rint(g[:, 7] * (SMALL_CAP * num_cells)).astype(np.int64)
            opp_value = np.rint(g[:, 8] * (SMALL_CAP * num_cells)).astype(np.int64)
            own_value = own_value + np.where(is_reinforce, points_left, 0)
            ratio = own_value.astype(np.float32) / np.maximum(1, opp_value).astype(np.float32)
            g[:, V103_VALUE_RATIO] = np.minimum(np.float32(V103_MAX_VALUE_RATIO), ratio)

            is_attack = (g[:, 0] == 1.0) | (g[:, 1] == 1.0)
            any_legal_non_pass = binaryInputNCHW[rows, 29].reshape(g.shape[0], -1).max(axis=1) > 0
            g[:, V103_CAN_ATTACK] = (is_attack & any_legal_non_pass).astype(np.float32)
            globalInputNC[rows] = g
    else:
        assert False, f"Unsupported model inputs version {model_inputs_version}"

def read_npz_training_data(
    npz_files,
    batch_size: int,
    world_size: int,
    rank: int,
    pos_len: int,
    device,
    randomize_symmetries: bool,
    model_config: modelconfigs.ModelConfig,
):
    rand = np.random.default_rng(seed=list(os.urandom(12)))
    num_bin_features = modelconfigs.get_num_bin_input_features(model_config)
    num_global_features = modelconfigs.get_num_global_input_features(model_config)

    for npz_file in npz_files:
        with np.load(npz_file) as npz:
            binaryInputNCHWPacked = npz["binaryInputNCHWPacked"]
            globalInputNC = npz["globalInputNC"]
            policyTargetsNCMove = npz["policyTargetsNCMove"].astype(np.float32)
            globalTargetsNC = npz["globalTargetsNC"]
            scoreDistrN = npz["scoreDistrN"].astype(np.float32)
            valueTargetsNCHW = npz["valueTargetsNCHW"].astype(np.float32)
        del npz

        binaryInputNCHW = np.unpackbits(binaryInputNCHWPacked,axis=2)
        assert len(binaryInputNCHW.shape) == 3
        assert binaryInputNCHW.shape[2] == ((pos_len * pos_len + 7) // 8) * 8
        binaryInputNCHW = binaryInputNCHW[:,:,:pos_len*pos_len]
        binaryInputNCHW = np.reshape(binaryInputNCHW, (
            binaryInputNCHW.shape[0], binaryInputNCHW.shape[1], pos_len, pos_len
        )).astype(np.float32)

        assert binaryInputNCHW.shape[1] == num_bin_features
        assert globalInputNC.shape[1] == num_global_features
        convert_inputs_for_model(
            binaryInputNCHW, globalInputNC, globalTargetsNC, modelconfigs.get_inputs_version(model_config)
        )

        num_samples = binaryInputNCHW.shape[0]
        # Just discard stuff that doesn't divide evenly
        num_whole_steps = num_samples // (batch_size * world_size)

        #logging.info(f"Beginning {npz_file} with {num_whole_steps * world_size} usable batches, my rank is {rank}")
        for n in range(num_whole_steps):
            start = (n * world_size + rank) * batch_size
            end = start + batch_size

            batch_binaryInputNCHW = torch.from_numpy(binaryInputNCHW[start:end]).to(device)
            batch_globalInputNC = torch.from_numpy(globalInputNC[start:end]).to(device)
            batch_policyTargetsNCMove = torch.from_numpy(policyTargetsNCMove[start:end]).to(device)
            batch_globalTargetsNC = torch.from_numpy(globalTargetsNC[start:end]).to(device)
            batch_scoreDistrN = torch.from_numpy(scoreDistrN[start:end]).to(device)
            batch_valueTargetsNCHW = torch.from_numpy(valueTargetsNCHW[start:end]).to(device)


            if randomize_symmetries:
                # Influence is played on a hex grid with neighbours (0,+-1),(+-1,0),(1,1),(-1,-1).
                # Only identity, rotate 180 (2), transpose (4) and anti-transpose (6) preserve it.
                symm = 2 * int(rand.integers(0,4))
                batch_binaryInputNCHW = apply_symmetry(batch_binaryInputNCHW, symm)
                batch_policyTargetsNCMove = apply_symmetry_policy(batch_policyTargetsNCMove, symm, pos_len)
                batch_valueTargetsNCHW = apply_symmetry(batch_valueTargetsNCHW, symm)
            batch_binaryInputNCHW = batch_binaryInputNCHW.contiguous()
            batch_policyTargetsNCMove = batch_policyTargetsNCMove.contiguous()
            batch_valueTargetsNCHW = batch_valueTargetsNCHW.contiguous()

            batch = dict(
                binaryInputNCHW = batch_binaryInputNCHW,
                globalInputNC = batch_globalInputNC,
                policyTargetsNCMove = batch_policyTargetsNCMove,
                globalTargetsNC = batch_globalTargetsNC,
                scoreDistrN = batch_scoreDistrN,
                valueTargetsNCHW = batch_valueTargetsNCHW,
            )
            yield batch


def apply_symmetry_policy(tensor, symm, pos_len):
    """Same as apply_symmetry but also handles the pass index"""
    batch_size = tensor.shape[0]
    channels = tensor.shape[1]
    tensor_without_pass = tensor[:,:,:-1].view((batch_size, channels, pos_len, pos_len))
    tensor_transformed = apply_symmetry(tensor_without_pass, symm)
    return torch.cat((
        tensor_transformed.reshape(batch_size, channels, pos_len*pos_len),
        tensor[:,:,-1:]
    ), dim=2)

def apply_symmetry(tensor, symm):
    """
    Apply a symmetry operation to the given tensor.

    Args:
        tensor (torch.Tensor): Tensor to be rotated. (..., W, W)
        symm (int):
            0, 1, 2, 3: Rotation by symm * pi / 2 radians.
            4, 5, 6, 7: Mirror symmetry on top of rotation.
    """
    assert tensor.shape[-1] == tensor.shape[-2]

    if symm == 0:
        return tensor
    if symm == 1:
        return tensor.transpose(-2, -1).flip(-2)
    if symm == 2:
        return tensor.flip(-1).flip(-2)
    if symm == 3:
        return tensor.transpose(-2, -1).flip(-1)
    if symm == 4:
        return tensor.transpose(-2, -1)
    if symm == 5:
        return tensor.flip(-1)
    if symm == 6:
        return tensor.transpose(-2, -1).flip(-1).flip(-2)
    if symm == 7:
        return tensor.flip(-2)

