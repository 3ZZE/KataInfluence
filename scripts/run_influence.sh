# Influence training loop: selfplay -> shuffle -> train -> export, repeated.
# Defaults are sized for a ~2 day run on one RTX 2080 Ti (see the notes at the bottom).

# Model
TRAINING_NAME=b10c128
MODEL_KIND=b10c128-fson-mish-rvglr-bnh
BATCH_SIZE=128
LR_SCALE=2.0

# Per generation. KataGomo guideline: ~100 trained samples per selfplay game,
# check that vloss_val - vloss_train < 0.05 (else more games or fewer samples).
GAMES_PER_GEN=3000   # >> numGameThreads (768), so the end-of-generation tail with few running games stays short
SAMPLES_PER_EPOCH=600000   # val vloss <= train vloss after 7 gens, each gen adds 0.6-0.9M rows
export KEEP_TARGET_ROWS=660000   # slightly above SAMPLES_PER_EPOCH
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
#   ~38 with 0.75 / 0.9). 3000 games per generation take ~1h of selfplay + ~6 min shuffle/train.
#   Generations get faster as games shorten.
# - If generations are much slower or faster, scale GAMES_PER_GEN and SAMPLES_PER_EPOCH together,
#   but keep GAMES_PER_GEN well above numGameThreads.
