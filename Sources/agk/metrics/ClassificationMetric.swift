public protocol ClassificationMetric {
    var name: String { get }
    func compute<Model, SourceElement>(model: Model, labels: [Int32], logits: [Float], dataset: KnowledgeGraphDataset<SourceElement, Int32>
    ) -> Float where Model: GenericModel, Model.Scalar == Float
    func aggregate(scores: [Float]) -> Float
    func nPositive(_ labels: [Int32]) -> Float
    func nMatching(_ trainLabels: [Int32], _ testLabels: [Int32], onlyPositive: Bool) -> Float
}
