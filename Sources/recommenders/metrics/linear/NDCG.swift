import TensorFlow
import Foundation

public struct NDCG: LinearMetric {
    let n: Int

    public init(n: Int) {
        self.n = n
    }

    public var name: String {
        "NDCG@\(n)"
    }

    public func compute<Model>(model: Model, trainFrame: TripleFrame, testFrame: TripleFrame, dataset: KnowledgeGraphDataset) -> Float where Model: GraphModel {
        func getDCG(_ degrees: [CorruptionDegree]) -> Float {
            degrees.enumerated().map { item in
                Float(item.element.rawValue) / log2(Float(item.offset) + 2)
            }.reduce(0, +)
        }

        var finalScores: [Float] = []
        for validFrame in testFrame.getCombinations(k: min(testFrame.data.count, n)) {
            let eitherHeadEitherTailCorruptedFrame = validFrame.sampleNegativeFrame(negativeFrame: dataset.normalizedNegativeFrame)
            let headAndTailCorruptedFrame = validFrame.sampleNegativeFrame(negativeFrame: dataset.normalizedNegativeFrame, corruptionDegree: CorruptionDegree.headAndTail)
            let completelyCorruptedFrame = validFrame.sampleNegativeFrame(negativeFrame: dataset.normalizedNegativeFrame, corruptionDegree: CorruptionDegree.complete)
            let totalTensor = Tensor(
                    stacking: validFrame.tensor.unstacked() + eitherHeadEitherTailCorruptedFrame.tensor.unstacked() +
                            headAndTailCorruptedFrame.tensor.unstacked() + completelyCorruptedFrame.tensor.unstacked()
            )
            let scores = model(totalTensor).unstacked().map {
                $0.scalarized()
            }
            let orderedCorruptionDegrees = Array(repeating: CorruptionDegree.none, count: validFrame.data.count) +
                    Array(repeating: CorruptionDegree.eitherHeadEitherTail, count: eitherHeadEitherTailCorruptedFrame.data.count) +
                    Array(repeating: CorruptionDegree.headAndTail, count: headAndTailCorruptedFrame.data.count) +
                    Array(repeating: CorruptionDegree.complete, count: completelyCorruptedFrame.data.count)
            let rankedCorruptionDegrees = orderedCorruptionDegrees.enumerated().sorted() { (lhs, rhs) in
                scores[lhs.offset] < scores[rhs.offset]
            }.map { item in
                item.element
            }
            finalScores.append(getDCG(rankedCorruptionDegrees) / getDCG(orderedCorruptionDegrees))
        }
        return aggregate(scores: finalScores)
    }
}
