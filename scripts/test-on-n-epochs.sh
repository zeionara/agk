swift build --product agk
for n_epochs in 10 50 100 500 1000 5000 10000; do
    for learning_rate in 0.001 0.01 0.1; do
        echo ""
        echo "Truncated GCN model (entity embeddings + output layer) trained $n_epochs epochs, learning-rate = $learning_rate"
        echo ""
        /home/zeio/agk/.build/debug/agk cross-validate -m $1 -d deduplicated-dataset.txt -n $n_epochs -l $learning_rate | grep -E "(Precision|Recall|F1Score|Accuracy)@[0-9]\.[0-9]+:"
    done
done
