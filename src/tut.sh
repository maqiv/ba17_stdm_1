#!/bin/sh

cat > params.ini << EOF
[Params]
ITERATIONS = 400000
BATCH_SIZE = 128
LAYERS = 2
LEARNING_RATE = 0.001
HIDDEN = 10
EOF
echo "running Training"
python lstm_multi_test.py
echo "trining finished"

cat > params.ini << EOF
[Params]
ITERATIONS = 400000
BATCH_SIZE = 128
LAYERS = 2
LEARNING_RATE = 0.001
HIDDEN = 25
EOF
echo "running Training"
python lstm_multi_test.py
echo "trining finished"


cat > params.ini << EOF
[Params]
ITERATIONS = 400000
BATCH_SIZE = 128
LAYERS = 2
LEARNING_RATE = 0.001
HIDDEN = 50
EOF
echo "running Training"
python lstm_multi_test.py
echo "trining finished"


cat > params.ini << EOF
[Params]
ITERATIONS = 400000
BATCH_SIZE = 128
LAYERS = 2
LEARNING_RATE = 0.001
HIDDEN = 100
EOF
echo "running Training"
python lstm_multi_test.py
echo "trining finished"

cat > params.ini << EOF
[Params]
ITERATIONS = 400000
BATCH_SIZE = 128
LAYERS = 3
LEARNING_RATE = 0.001
HIDDEN = 128
EOF
echo "running Training"
python lstm_multi_test.py
echo "trining finished"

cat > params.ini << EOF
[Params]
ITERATIONS = 400000
BATCH_SIZE = 128
LAYERS = 4
LEARNING_RATE = 0.001
HIDDEN = 128
EOF
echo "running Training"
python lstm_multi_test.py
echo "trining finished"

