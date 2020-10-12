import TensorFlow

public struct MRR: LinearMetric {
    let n: Int

    public init(n: Int) {
        self.n = n
    }

    public var name: String {
        "MRR@\(n)"
    }

    public func compute<Model>(model: Model, trainFrame: TripleFrame, testFrame: TripleFrame, dataset: KnowledgeGraphDataset) -> Float where Model: GraphModel {
        func getSortedTriples(validFrame: TripleFrame, corruptedFrame: TripleFrame, scores: [Float]) -> [[Int32]] {
            (validFrame.data + corruptedFrame.data).enumerated().sorted() { (lhs, rhs) in
                scores[lhs.offset] < scores[rhs.offset]
            }.map { (i: Int, triple: [Int32]) in
                triple
            }
        }

        var finalScores: [Float] = []
        for validFrame in testFrame.getCombinations(k: min(testFrame.data.count, n)) {
            let corruptedFrame = validFrame.sampleNegativeFrame(negativeFrame: dataset.normalizedNegativeFrame)
            let totalTensor = Tensor(stacking: validFrame.tensor.unstacked() + corruptedFrame.tensor.unstacked())
            let scores = model(totalTensor).unstacked().map {
                $0.scalarized()
            }
            finalScores.append(
                    Float(1.0) / Float(
                            getSortedTriples(validFrame: validFrame, corruptedFrame: corruptedFrame, scores: scores).enumerated().filter { item in
                                validFrame.data.contains(item.element)
                            }.first!.offset + 1
                    )
            )
        }
        return aggregate(scores: finalScores)
    }
}
