set -o pipefail
cd "$(dirname "$0")"

RUN_NAME="${1:-run1}"
RUN_DIR="./$RUN_NAME"

if [[ -e "$RUN_DIR" ]]; then
    echo "Error: $RUN_DIR already exists" >&2
    exit 1
fi

mkdir -p "$RUN_DIR/engine"
cp ../cpp/katago "$RUN_DIR/engine/"
chmod +x "$RUN_DIR/engine/katago"

mkdir -p "$RUN_DIR/train"
cp ../python/*.py ../python/*.sh "$RUN_DIR/train/"

cp ./run_influence.sh   "$RUN_DIR/run_influence.sh"
cp ./selfplay.cfg  "$RUN_DIR/selfplay.cfg"
chmod +x "$RUN_DIR/run_influence.sh"

echo "Created $RUN_DIR. Run: cd $RUN_DIR && bash run_influence.sh"
