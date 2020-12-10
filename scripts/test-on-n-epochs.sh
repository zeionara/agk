test() {
    echo ""
    echo "gcn-truncated(epochs: $2, learning-rate: $3)"
    echo ""
    /home/zeio/agk/.build/debug/agk cross-validate -m $1 -d deduplicated-dataset.txt -n $2 -l $3 | grep -E "(Precision|Recall|F1Score|Accuracy)@[0-9]\.[0-9]+.*:"
}


swift build --product agk
for n_epochs in 10 50 100 500 1000 5000 10000; do
    for learning_rate in 0.01 0.015 0.07; do
        test $1 $n_epochs $learning_rate
    done
done
