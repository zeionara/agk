# echo ""
# echo 'transe(margin: 1.0, learning-rate: 0.01, loss: sum, optimizer: adam)'
# echo ""
# ./.build/debug/agk cross-validate -m transe -d deduplicated-dataset.txt --margin 1.0 -l 0.01 --linear-model-loss sum --optimizer adam
# echo ""
# echo 'transe(margin: 1.0, learning-rate: 0.01, loss: sigmoid, optimizer: adam)'
# echo ""
# ./.build/debug/agk cross-validate -m transe -d deduplicated-dataset.txt --margin 1.0 -l 0.01 --linear-model-loss sigmoid --optimizer adam
# echo ""
# echo 'transe(margin: 1.0, learning-rate: 0.03, loss: sum, optimizer: adam)'
# echo ""
# ./.build/debug/agk cross-validate -m transe -d deduplicated-dataset.txt --margin 1.0 -l 0.03 --linear-model-loss sum --optimizer adam
# echo ""
# echo 'transe(margin: 1.0, learning-rate: 0.03, loss: sigmoid, optimizer: adam)'
# echo ""
# ./.build/debug/agk cross-validate -m transe -d deduplicated-dataset.txt --margin 1.0 -l 0.03 --linear-model-loss sigmoid --optimizer adam
# echo ""
# echo 'transe(margin: 2.0, learning-rate: 0.01, loss: sum, optimizer: adam)'
# echo ""
# ./.build/debug/agk cross-validate -m transe -d deduplicated-dataset.txt --margin 2.0 -l 0.01 --linear-model-loss sum --optimizer adam
# echo ""
# echo 'transe(margin: 2.0, learning-rate: 0.01, loss: sigmoid, optimizer: adam)'
# echo ""
# ./.build/debug/agk cross-validate -m transe -d deduplicated-dataset.txt --margin 2.0 -l 0.01 --linear-model-loss sigmoid --optimizer adam
# echo ""
# echo 'transe(margin: 2.0, learning-rate: 0.03, loss: sum, optimizer: adam)'
# echo ""
# ./.build/debug/agk cross-validate -m transe -d deduplicated-dataset.txt --margin 2.0 -l 0.03 --linear-model-loss sum --optimizer adam
# echo ""
# echo 'transe(margin: 2.0, learning-rate: 0.03, loss: sigmoid, optimizer: adam)'
# echo ""
# ./.build/debug/agk cross-validate -m transe -d deduplicated-dataset.txt --margin 2.0 -l 0.03 --linear-model-loss sigmoid --optimizer adam
# echo ""
# echo 'transe(margin: 3.0, learning-rate: 0.01, loss: sum, optimizer: adam)'
# echo ""
# ./.build/debug/agk cross-validate -m transe -d deduplicated-dataset.txt --margin 3.0 -l 0.01 --linear-model-loss sum --optimizer adam
# echo ""
# echo 'transe(margin: 3.0, learning-rate: 0.01, loss: sigmoid, optimizer: adam)'
# echo ""
# ./.build/debug/agk cross-validate -m transe -d deduplicated-dataset.txt --margin 3.0 -l 0.01 --linear-model-loss sigmoid --optimizer adam
# echo ""
# echo 'transe(margin: 3.0, learning-rate: 0.03, loss: sum, optimizer: adam)'
# echo ""
# ./.build/debug/agk cross-validate -m transe -d deduplicated-dataset.txt --margin 3.0 -l 0.03 --linear-model-loss sum --optimizer adam
# echo ""
# echo 'transe(margin: 3.0, learning-rate: 0.03, loss: sigmoid, optimizer: adam)'
# echo ""
# ./.build/debug/agk cross-validate -m transe -d deduplicated-dataset.txt --margin 3.0 -l 0.03 --linear-model-loss sigmoid --optimizer adam
# echo ""
# echo 'transd(margin: 1.0, learning-rate: 0.01, loss: sum, optimizer: adam)'
# echo ""
# ./.build/debug/agk cross-validate -m transd -d deduplicated-dataset.txt --margin 1.0 -l 0.01 --linear-model-loss sum --optimizer adam
# echo ""
# echo 'transd(margin: 1.0, learning-rate: 0.01, loss: sigmoid, optimizer: adam)'
# echo ""
# ./.build/debug/agk cross-validate -m transd -d deduplicated-dataset.txt --margin 1.0 -l 0.01 --linear-model-loss sigmoid --optimizer adam
# echo ""
# echo 'transd(margin: 1.0, learning-rate: 0.03, loss: sum, optimizer: adam)'
# echo ""
# ./.build/debug/agk cross-validate -m transd -d deduplicated-dataset.txt --margin 1.0 -l 0.03 --linear-model-loss sum --optimizer adam
echo ""
echo 'transd(margin: 1.0, learning-rate: 0.03, loss: sigmoid, optimizer: adam)'
echo ""
./.build/debug/agk cross-validate -m transd -d deduplicated-dataset.txt --margin 1.0 -l 0.03 --linear-model-loss sigmoid --optimizer adam
echo ""
echo 'transd(margin: 2.0, learning-rate: 0.01, loss: sum, optimizer: adam)'
echo ""
./.build/debug/agk cross-validate -m transd -d deduplicated-dataset.txt --margin 2.0 -l 0.01 --linear-model-loss sum --optimizer adam
echo ""
echo 'transd(margin: 2.0, learning-rate: 0.01, loss: sigmoid, optimizer: adam)'
echo ""
./.build/debug/agk cross-validate -m transd -d deduplicated-dataset.txt --margin 2.0 -l 0.01 --linear-model-loss sigmoid --optimizer adam
echo ""
echo 'transd(margin: 2.0, learning-rate: 0.03, loss: sum, optimizer: adam)'
echo ""
./.build/debug/agk cross-validate -m transd -d deduplicated-dataset.txt --margin 2.0 -l 0.03 --linear-model-loss sum --optimizer adam
echo ""
echo 'transd(margin: 2.0, learning-rate: 0.03, loss: sigmoid, optimizer: adam)'
echo ""
./.build/debug/agk cross-validate -m transd -d deduplicated-dataset.txt --margin 2.0 -l 0.03 --linear-model-loss sigmoid --optimizer adam
echo ""
echo 'transd(margin: 3.0, learning-rate: 0.01, loss: sum, optimizer: adam)'
echo ""
./.build/debug/agk cross-validate -m transd -d deduplicated-dataset.txt --margin 3.0 -l 0.01 --linear-model-loss sum --optimizer adam
echo ""
echo 'transd(margin: 3.0, learning-rate: 0.01, loss: sigmoid, optimizer: adam)'
echo ""
./.build/debug/agk cross-validate -m transd -d deduplicated-dataset.txt --margin 3.0 -l 0.01 --linear-model-loss sigmoid --optimizer adam
echo ""
echo 'transd(margin: 3.0, learning-rate: 0.03, loss: sum, optimizer: adam)'
echo ""
./.build/debug/agk cross-validate -m transd -d deduplicated-dataset.txt --margin 3.0 -l 0.03 --linear-model-loss sum --optimizer adam
echo ""
echo 'transd(margin: 3.0, learning-rate: 0.03, loss: sigmoid, optimizer: adam)'
echo ""
./.build/debug/agk cross-validate -m transd -d deduplicated-dataset.txt --margin 3.0 -l 0.03 --linear-model-loss sigmoid --optimizer adam
echo ""
echo 'rotate(margin: 1.0, learning-rate: 0.01, loss: sum, optimizer: adam)'
echo ""
./.build/debug/agk cross-validate -m rotate -d deduplicated-dataset.txt --margin 1.0 -l 0.01 --linear-model-loss sum --optimizer adam
echo ""
echo 'rotate(margin: 1.0, learning-rate: 0.01, loss: sigmoid, optimizer: adam)'
echo ""
./.build/debug/agk cross-validate -m rotate -d deduplicated-dataset.txt --margin 1.0 -l 0.01 --linear-model-loss sigmoid --optimizer adam
echo ""
echo 'rotate(margin: 1.0, learning-rate: 0.03, loss: sum, optimizer: adam)'
echo ""
./.build/debug/agk cross-validate -m rotate -d deduplicated-dataset.txt --margin 1.0 -l 0.03 --linear-model-loss sum --optimizer adam
echo ""
echo 'rotate(margin: 1.0, learning-rate: 0.03, loss: sigmoid, optimizer: adam)'
echo ""
./.build/debug/agk cross-validate -m rotate -d deduplicated-dataset.txt --margin 1.0 -l 0.03 --linear-model-loss sigmoid --optimizer adam
echo ""
echo 'rotate(margin: 2.0, learning-rate: 0.01, loss: sum, optimizer: adam)'
echo ""
./.build/debug/agk cross-validate -m rotate -d deduplicated-dataset.txt --margin 2.0 -l 0.01 --linear-model-loss sum --optimizer adam
echo ""
echo 'rotate(margin: 2.0, learning-rate: 0.01, loss: sigmoid, optimizer: adam)'
echo ""
./.build/debug/agk cross-validate -m rotate -d deduplicated-dataset.txt --margin 2.0 -l 0.01 --linear-model-loss sigmoid --optimizer adam
echo ""
echo 'rotate(margin: 2.0, learning-rate: 0.03, loss: sum, optimizer: adam)'
echo ""
./.build/debug/agk cross-validate -m rotate -d deduplicated-dataset.txt --margin 2.0 -l 0.03 --linear-model-loss sum --optimizer adam
echo ""
echo 'rotate(margin: 2.0, learning-rate: 0.03, loss: sigmoid, optimizer: adam)'
echo ""
./.build/debug/agk cross-validate -m rotate -d deduplicated-dataset.txt --margin 2.0 -l 0.03 --linear-model-loss sigmoid --optimizer adam
echo ""
echo 'rotate(margin: 3.0, learning-rate: 0.01, loss: sum, optimizer: adam)'
echo ""
./.build/debug/agk cross-validate -m rotate -d deduplicated-dataset.txt --margin 3.0 -l 0.01 --linear-model-loss sum --optimizer adam
echo ""
echo 'rotate(margin: 3.0, learning-rate: 0.01, loss: sigmoid, optimizer: adam)'
echo ""
./.build/debug/agk cross-validate -m rotate -d deduplicated-dataset.txt --margin 3.0 -l 0.01 --linear-model-loss sigmoid --optimizer adam
echo ""
echo 'rotate(margin: 3.0, learning-rate: 0.03, loss: sum, optimizer: adam)'
echo ""
./.build/debug/agk cross-validate -m rotate -d deduplicated-dataset.txt --margin 3.0 -l 0.03 --linear-model-loss sum --optimizer adam
echo ""
echo 'rotate(margin: 3.0, learning-rate: 0.03, loss: sigmoid, optimizer: adam)'
echo ""
./.build/debug/agk cross-validate -m rotate -d deduplicated-dataset.txt --margin 3.0 -l 0.03 --linear-model-loss sigmoid --optimizer adam
