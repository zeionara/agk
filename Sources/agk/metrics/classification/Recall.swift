import TensorFlow

public struct Recall: ClassificationMetric {
    public let threshold: Float
    private let nDecimalPlaces: Int

    public init(_ threshold: Float = 0.5, nDecimalPlaces: Int = 3) {
        self.threshold = threshold
        self.nDecimalPlaces = nDecimalPlaces
     }

    public var name: String {
        "Recall@" + String(format: "%.\(nDecimalPlaces)f", threshold)
    }

    public func compute<Model, SourceElement>(
            model: Model,
            labels: [Int32], logits: [Float], dataset: KnowledgeGraphDataset<SourceElement, Int32>
    ) -> Float where Model: GenericModel, Model.Scalar == Float {
        // print("logits: \(logits)")
        let testLabels = logits.map{$0 >= threshold ? 1 : 0}.map{Int32($0)}
        // print("logits: \(testLabels)")
        let divisor = nPositive(labels)
        return divisor > 0 ? nMatching(labels, testLabels) / divisor : Float.nan
    }
}
