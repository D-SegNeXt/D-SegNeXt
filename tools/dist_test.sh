CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4}

#python ./tools/test.py '../logs/20230913_140414/segnext.tiny.1024x1024.OPM.20k.py' '../logs/20230913_140414/latest.pth' --show-dir='../logs/20230913_140414/result/'
#bash ./tools/dist_test.sh '../logs/20230913_140414/segnext.tiny.512x512.OPM.20k.py' '../logs/20230913_140414/latest.pth' 2 --show-dir='../logs/20230913_140414/result/'