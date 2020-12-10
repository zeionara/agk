import TensorFlow

public struct F1Score: ClassificationMetric {
    public let threshold: Float
    private let nDecimalPlaces: Int
    private let precision: Precision
    private let recall: Recall
    private let reverse: Bool

    public init(_ threshold: Float = 0.5, nDecimalPlaces: Int = 3, reverse: Bool = false) {
        self.threshold = threshold
        self.nDecimalPlaces = nDecimalPlaces
        self.precision = Precision(threshold, reverse: reverse)
        self.recall = Recall(threshold, reverse: reverse)
        self.reverse = reverse
     }

    public var name: String {
        "F1Score@" + String(format: "%.\(nDecimalPlaces)f", threshold) + (reverse ? " (reversed)" : "")
    }

    public func compute<Model, SourceElement>(
            model: Model,
            labels: [Int32], logits: [Float], dataset: KnowledgeGraphDataset<SourceElement, Int32>
    ) -> Float where Model: GenericModel, Model.Scalar == Float {
        let precision = self.precision.compute(model: model, labels: labels, logits: logits, dataset: dataset)
        let recall = self.recall.compute(model: model, labels: labels, logits: logits, dataset: dataset)
        let divisor = precision + recall
        return divisor > 0 ? 2 * precision * recall / divisor : Float.nan
    }
}
