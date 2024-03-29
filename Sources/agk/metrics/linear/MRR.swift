import TensorFlow

public struct MRR: LinearMetric {
    let n: Int

    public init(n: Int) {
        self.n = n
    }

    public var name: String {
        "MRR@\(n)"
    }

    public func compute<Model, SourceElement>(
            model: Model,
            trainFrame: TripleFrame<Int32>, testFrame: TripleFrame<Int32>,
            dataset: KnowledgeGraphDataset<SourceElement, Int32>
    ) -> Float where Model: GenericModel, Model.Scalar == Int32 {
        func getSortedTriples(validFrame: TripleFrame<Int32>, corruptedFrame: TripleFrame<Int32>, scores: [Float]) -> [[Int32]] {
            (validFrame.data + corruptedFrame.data).enumerated().sorted() { (lhs, rhs) in
                scores[lhs.offset] < scores[rhs.offset]
            }.map { (i: Int, triple: [Int32]) in
                triple
            }
        }

        var finalScores: [Float] = []
        for validFrame in testFrame.getCombinations(k: min(testFrame.data.count, n)) {
//            print("Handling validation frame")
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
