# Influence training loop: selfplay -> shuffle -> train -> export, repeated.
# Defaults are sized for a ~2 day run on one RTX 2080 Ti (see the notes at the bottom).

# Model
TRAINING_NAME=b10c256nbt
MODEL_KIND=b10c256nbt-fson-mish-rvglr-bnh
BATCH_SIZE=256   # 128 left b10c256nbt training CPU-bound (1 core 100%, GPU 57%); 256 is ~35% faster
LR_SCALE=1.0

# Per generation. KataGomo guideline: ~100 trained samples per selfplay game,
# check that vloss_val - vloss_train < 0.05 (else more games or fewer samples).
GAMES_PER_GEN=6000   # >> numGameThreads (512), so the end-of-generation tail with few running games stays short
SAMPLES_PER_EPOCH=1500000   # 6000 games write ~1.1M rows, each trained on ~1.4 times (4000 games / 1.0M had val vloss below train)
export KEEP_TARGET_ROWS=1650000   # ~10% above SAMPLES_PER_EPOCH, so train.py (whole files of ~48k rows) never repeats a file within a gen
export MIN_ROWS=150000

mkdir -p data/selfplay/ data/models/

. ../.venv/bin/activate

while true
do
    chmod +x ./engine/katago  #linux
    # SKIP_FIRST_SELFPLAY=1 bash run_influence.sh: resume straight into shuffle/train (e.g. after a crash)
    if [[ -z "${SKIP_FIRST_SELFPLAY:-}" ]]; then
        ./engine/katago selfplay -models-dir data/models -config selfplay.cfg -output-dir data/selfplay -max-games-total "$GAMES_PER_GEN"
    fi
    SKIP_FIRST_SELFPLAY=
    cd train
    bash shuffle.sh ../data ./ktmp 16 "$BATCH_SIZE"
    bash train.sh ../data "$TRAINING_NAME" "$MODEL_KIND" "$BATCH_SIZE" main -lr-scale "$LR_SCALE" -samples-per-epoch "$SAMPLES_PER_EPOCH"
    bash export.sh influence ../data 0
    python view_loss.py
    cd ..
done

# Sizing notes (2026-10, measured on the 2080 Ti VPS, CUDA backend, b10c128, 1st trained net):
# - selfplay.cfg 768 game threads / batch 384 / 2 NN server threads: ~36k evals/s
#   (128/128/1 gave 22k). Fits in 16 GB RAM only with COMPILE_MAX_BOARD_LEN 11.
# - Games are ~2k moves; a move costs ~30 evals (cheapSearchProb 0.85 / reinforceCheapSearchProb 0.95,
#   ~38 with 0.75 / 0.9). 5000 games per generation take ~1.5h of selfplay + ~17 min shuffle/train (1.65M samples).
#   Generations get faster as games shorten.
# - If generations are much slower or faster, scale GAMES_PER_GEN and SAMPLES_PER_EPOCH together,
#   but keep GAMES_PER_GEN well above numGameThreads.
