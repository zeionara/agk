
import Datasets
import Foundation
import RecommendationModels
import TensorFlow

func testNeuMF(size: [Int], regs: [Float], numLatentFeatures: Int, matrixRegularization: Float, learningRate: Float, nTrainEpochs: Int){
    let dataset = MovieLens(trainBatchSize: 1024)
    let numUsers = dataset.numUsers
    let numItems = dataset.numItems

    var model = NeuMF(numUsers: numUsers, numItems: numItems, numLatentFeatures: numLatentFeatures, matrixRegularization: matrixRegularization, mlpLayerSizes: size, mlpRegularizations: regs)
    let optimizer = Adam(for: model, learningRate: learningRate)
    var itemCount = Dictionary(
            uniqueKeysWithValues: zip(
                    dataset.testUsers, Array(repeating: 0.0, count: dataset.testUsers.count)
            )
    )
    var testNegSampling = Tensor<Float>(zeros: [numUsers, numItems])

    for element in dataset.testData {
        let rating = element[2]
        if rating > 0 && dataset.item2id[element[1]] != nil {
            let uIndex = dataset.user2id[element[0]]!
            let iIndex = dataset.item2id[element[1]]!
            testNegSampling[uIndex][iIndex] = Tensor(1.0)
            itemCount[element[0]] = itemCount[element[0]]! + 1.0
        }
    }
    print("Dataset acquired.")

    print("Starting training...")
    let epochCount = nTrainEpochs
    for (epoch, epochBatches) in dataset.training.prefix(epochCount).enumerated() {
        var avgLoss: Float = 0.0
        Context.local.learningPhase = .training
        for batch in epochBatches {
            let userId = batch.first
            let rating = batch.second
            let (loss, grad) = valueWithGradient(at: model) { model -> Tensor<Float> in
                let logits = model(userId)
                return sigmoidCrossEntropy(logits: logits, labels: rating)
            }

            optimizer.update(&model, along: grad)
            avgLoss = avgLoss + loss.scalarized()
        }

        Context.local.learningPhase = .inference
        var correct = 0.0
        var count = 0
        for user in dataset.testUsers[0...30] {
            var negativeItem: [Float] = []
            var output: [Float] = []
            let userIndex = dataset.user2id[user]!
            for item in dataset.items {
                let itemIndex = dataset.item2id[item]!
                if dataset.trainNegSampling[userIndex][itemIndex].scalarized() == 0 {
                    let input = Tensor<Int32>(
                            shape: [1, 2], scalars: [Int32(userIndex), Int32(itemIndex)])
                    output.append(model(input).scalarized())
                    negativeItem.append(item)
                }
            }
            let itemScore = Dictionary(uniqueKeysWithValues: zip(negativeItem, output))
            let sortedItemScore = itemScore.sorted { $0.1 > $1.1 }
            let topK = sortedItemScore.prefix(min(10, Int(itemCount[user]!)))

            for (key, _) in topK {
                if testNegSampling[userIndex][dataset.item2id[key]!] == Tensor(1.0) {
                    correct = correct + 1.0
                }
                count = count + 1
            }
        }
        print(
                "Epoch: \(epoch)", "Current loss: \(avgLoss/1024.0)", "Validation Accuracy:",
                correct / Double(count))
    }

    print("Starting testing...")
    Context.local.learningPhase = .inference
    var correct = 0.0
    var count = 0
    for user in dataset.testUsers {
        var negativeItem: [Float] = []
        var output: [Float] = []
        let userIndex = dataset.user2id[user]!
        for item in dataset.items {
            let itemIndex = dataset.item2id[item]!
            if dataset.trainNegSampling[userIndex][itemIndex].scalarized() == 0 {
                let input = Tensor<Int32>(
                        shape: [1, 2], scalars: [Int32(userIndex), Int32(itemIndex)])
                output.append(model(input).scalarized())
                negativeItem.append(item)
            }
        }

        let itemScore = Dictionary(uniqueKeysWithValues: zip(negativeItem, output))
        let sortedItemScore = itemScore.sorted { $0.1 > $1.1 }
        let topK = sortedItemScore.prefix(min(10, Int(itemCount[user]!)))

        print("User:", user, terminator: "\t")
        print("Top K Recommended Items:", terminator: "\t")

        for (key, _) in topK {
            print(key, terminator: "\t")
            if testNegSampling[userIndex][dataset.item2id[key]!] == Tensor(1.0) {
                correct = correct + 1.0
            }
            count = count + 1
        }
        print(terminator: "\n")
    }
    print("Test Accuracy:", correct / Double(count))
}

