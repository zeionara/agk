public protocol ClassificationMetric {
    var name: String { get }
    func compute<Model, SourceElement>(model: Model, labels: [Int32], logits: [Float], dataset: KnowledgeGraphDataset<SourceElement, Int32>
    ) -> Float where Model: GenericModel
    func aggregate(scores: [Float]) -> Float
    func nPositive(_ labels: [Int32], reverse: Bool) -> Float
    func nMatching(_ trainLabels: [Int32], _ testLabels: [Int32], onlyPositive: Bool, onlyNegative: Bool) -> Float
}

extension Bool {
    var asInt: Int32 {
        return self ? 1 : 0
    }
}