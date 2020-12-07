import Foundation

public extension LinearMetric {
    func aggregate(scores: [Float]) -> Float {
        scores.reduce(0.0, +) / Float(scores.count)
    }
}
