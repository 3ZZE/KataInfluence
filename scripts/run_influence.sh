mkdir data/
mkdir data/selfplay/
mkdir data/models/


. ../.venv/bin/activate

while true
do 
    chmod +x ./engine/katago  #linux
    ./run_selfplay.sh
    cd train
    bash shuffle.sh ../data ./ktmp 16 128
    bash train.sh ../data b10c256n b10c256nbt-fson-rvgl-bnh 128 main -lr-scale 1  -samples-per-epoch 1000000
    bash export.sh test ../data 0
    python view_loss.py
    cd ..
done
