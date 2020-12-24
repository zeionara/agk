DELAY=5 # seconds

test() {
    echo ""
    echo "$1(margin: $2, learning-rate: $3, loss: $4, optimizer: $5)"
    echo ""
    COMMAND="./.build/debug/agk cross-validate -m $1 -d deduplicated-dataset.txt --margin $2 -l $3 --linear-model-loss $4 --optimizer $5"
    echo " trace Running with command: '$COMMAND'"
    eval $COMMAND
}

for model in transe transd rotate; do
    for margin in 1.0 2.0 3.0; do
        for learning_rate in 0.01 0.03; do
            for loss in sum sigmoid; do
                for optimizer in adam; do
                    test $model $margin $learning_rate $loss $optimizer
                    sleep $DELAY
                done
            done
        done
    done
done
