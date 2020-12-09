swift build --product agk
for n_epochs in 10, 50, 100, 500, 1000, 5000, 10000; do
    /home/zeio/agk/.build/debug/agk cross-validate -m $1 -d deduplicated-dataset.txt
done
