import TensorFlow

public struct Precision: ClassificationMetric {
    public let threshold: Float
    private let nDecimalPlaces: Int
    private let reverse: Bool

    public init(_ threshold: Float = 0.5, nDecimalPlaces: Int = 3, reverse: Bool = false) {
        self.threshold = threshold
        self.nDecimalPlaces = nDecimalPlaces
        self.reverse = reverse
     }

    public var name: String {
        "Precision@" + String(format: "%.\(nDecimalPlaces)f", threshold) + (reverse ? " (reversed)" : "")
    }

    public func compute<Model, SourceElement>(
            model: Model,
            labels: [Int32], logits: [Float], dataset: KnowledgeGraphDataset<SourceElement, Int32>
    ) -> Float where Model: GenericModel {
        // print("logits: \(logits)")
        // print(logits)
        let testLabels = logits.map{$0 >= threshold ? (!reverse).asInt : reverse.asInt}.map{Int32($0)}
        // print(testLabels)
        // print("logits: \(testLabels)")
        let divisor = nPositive(testLabels, reverse: reverse)
        // print(divisor)
        return divisor > 0 ? nMatching(labels, testLabels, onlyPositive: !reverse, onlyNegative: reverse) / divisor : Float.nan
    }
}
