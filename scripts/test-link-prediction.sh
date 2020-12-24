REPORT_NAME=${1:-link-prediction}
RAW_REPORT_PATH=$PWD/data/raw-reports/$REPORT_NAME.txt
TMP_RAW_REPORT_PATH=$PWD/data/raw-reports/$REPORT_NAME-tmp.txt

# 1. Compile project

swift build --product agk
echo Compilation was finished

# 2. Perform experiments and generate metrics

./scripts/run-link-prediction-experiments.sh |& tee $TMP_RAW_REPORT_PATH
cat $TMP_RAW_REPORT_PATH | grep -v " trace " > $RAW_REPORT_PATH
rm $TMP_RAW_REPORT_PATH
sed -E -i "s/.+cv-tester : //g" $RAW_REPORT_PATH
# | grep --line-buffered -E "(MRR|Hits|MAP|NDCG)@[0-9].*:"
# | sed -E "s/.+cv-tester : //g"
# | grep --line-buffered -E "(MRR|Hits|MAP|NDCG)@[0-9].*:"
# echo "" > $RAW_REPORT_PATH
# echo Experiments were completed

# 3. Generate report

./.build/debug/agk restructure-report raw-reports/$REPORT_NAME.txt reports/$REPORT_NAME.txt
echo Report was generated
