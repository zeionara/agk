extension ClassificationMetric {
    public func aggregate(scores: [Float]) -> Float {
        scores.reduce(0.0, +) / Float(scores.count)
    }

    public func nPositive(_ labels: [Int32], reverse: Bool = false) -> Float {
        Float(labels.filter{$0 == (reverse ? 0 : 1)}.count)
    }

    public func nMatching(_ trainLabels: [Int32], _ testLabels: [Int32], onlyPositive: Bool = true, onlyNegative: Bool = false) -> Float {
        func doMatch(trainLabel: Int32, testLabel: Int32) -> Bool {
            return ((((trainLabel == 1) || !onlyPositive) && ((trainLabel == 0) || !onlyNegative_)) && (trainLabel == testLabel))
        }
        var onlyNegative_ = onlyNegative
        if onlyPositive && onlyNegative {
            onlyNegative_ = false
        }
        return Float(zip(trainLabels, testLabels).filter{(trainLabel, testLabel) in doMatch(trainLabel: trainLabel, testLabel: testLabel)}.count)
    }
}
