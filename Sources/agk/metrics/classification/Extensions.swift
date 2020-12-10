extension ClassificationMetric {
    public func aggregate(scores: [Float]) -> Float {
        scores.reduce(0.0, +) / Float(scores.count)
    }

    public func nPositive(_ labels: [Int32]) -> Float {
        labels.filter{$0 == 1}.map{Float($0)}.reduce(0.0, +)
    }

    public func nMatching(_ trainLabels: [Int32], _ testLabels: [Int32], onlyPositive: Bool = true) -> Float {
        zip(trainLabels, testLabels).map{(trainLabel, testLabel) in ((trainLabel == 1) || !onlyPositive) && (trainLabel == testLabel) ? 1.0 : 0.0}.reduce(0.0, +)
    }
}
