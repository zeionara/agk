import TensorFlow

public struct Precision: ClassificationMetric {
    public init() { }

    public var name: String {
        "Precision"
    }

    public func compute<Model, SourceElement>(
            model: Model,
            trainLabels: LabelFrame<Int32>, testLabels: LabelFrame<Int32>, dataset: KnowledgeGraphDataset<SourceElement, Int32>
    ) -> Float where Model: GenericModel, Model.Scalar == Float {
        let divisor = trainLabels.labels.unstacked().map{$0.scalar!}.filter{$0 == 1}.reduce(0.0, +)
        return divisor > 0 ? zip(trainLabels.labels.unstacked(), testLabels.labels.unstacked()).map{(trainLabel, testLabel) in trainLabel.scalar == testLabel.scalar ? 1.0 : 0.0}.reduce(0.0, +) / divisor : 1.0
    }

    public func aggregate(scores: [Float]) -> Float {
        scores.reduce(0.0, +) / Float(scores.count)
    }
}
