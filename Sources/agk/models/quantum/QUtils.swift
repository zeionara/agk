import Foundation
import TensorFlow

public func computeL2Norm(data: [Double]) -> Double {
    return sqrt(data.reduce(0, {x, y in x + pow(y, 2)}))
}

public func normalizeWithL2(vector: [Double]) -> [Double] {
    let norm = computeL2Norm(data: vector)
    return vector.map{$0 / norm}
}

public extension Array where Element == Double {
    var normalized: [Double] {
        normalizeWithL2(vector: self)
    }

    var asProbabilities: [Double] {
        var result = [Double]()
        for item in normalized {
            result.append(Double.pow(item, 2))
        }
        return result
    }
}

public func computeTensorL2Norm(data: Tensor<Double>) -> Tensor<Double> {
    sqrt((data * data).sum(alongAxes: [1]))
}

public func normalizeTensorWithL2(tensor: Tensor<Double>) -> Tensor<Double> {
    tensor / computeTensorL2Norm(data: tensor)
}

public func initQuantumEntityEmbeddings(dimensionality: Int, nItems: Int, device device_: Device) -> Embedding<Double> {
    Embedding(
            embeddings: normalizeTensorWithL2(
                    tensor: Tensor<Double>(
                            randomUniform: [nItems, dimensionality],
                            lowerBound: Tensor(Double(0.0), on: device_),
                            upperBound: Tensor(Double(1.0) / Double(dimensionality), on: device_),
                            on: device_
                    )
            )
    )
}