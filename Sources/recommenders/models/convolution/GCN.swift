import Foundation
import TensorFlow

//private func computeL2Norm(data: Tensor<Float>) -> Tensor<Float> {
//    sqrt((data * data).sum(alongAxes: [1]))
//}
//
//private func normalizeWithL2(tensor: Tensor<Float>) -> Tensor<Float> {
//    tensor / computeL2Norm(data: tensor)
//}

private func computeScore(head: Tensor<Float>, tail: Tensor<Float>, relationship: Tensor<Float>) -> Tensor<Float> {
//    let normalizedHead = normalizeWithL2(tensor: head)
//    let normalizedTail = normalizeWithL2(tensor: tail)
//    let normalizedRelationship = normalizeWithL2(tensor: relationship)
    let score = head + (relationship - tail)
    let norma = computeL2Norm(data: score)

    return norma
}

public struct GCN: GraphModel {
    public var entityEmbeddings: Embedding<Float>
    public var outputLayer: Tensor<Float>
    @noDerivative
    public let device: Device


    public init(embeddingDimensionality: Int = 100, dataset: KnowledgeGraphDataset? = Optional.none, device device_: Device = Device.default, entityEmbeddings: Embedding<Float>? = Optional.none) {
        if let entityEmbeddings_ = entityEmbeddings {
            self.entityEmbeddings = entityEmbeddings_
        } else {
            self.entityEmbeddings = initEmbeddings(dimensionality: embeddingDimensionality, nItems: dataset!.frame.entities.count + dataset!.frame.relationships.count * 2, device: device_)
        }
        outputLayer = Tensor<Float>(
                randomUniform: [embeddingDimensionality, 1],
                lowerBound: Tensor(Float(-1.0) / Float(embeddingDimensionality), on: device_),
                upperBound: Tensor(Float(1.0) / Float(embeddingDimensionality), on: device_),
                on: device_
        )
        device = device_
    }

    public func normalizeEmbeddings() -> GCN {
        GCN(
                embeddingDimensionality: 100,
                device: device,
                entityEmbeddings: Embedding(embeddings: normalizeWithL2(tensor: entityEmbeddings.embeddings))
        )
    }

    @differentiable
    public func callAsFunction(_ matrix: Tensor<Int32>) -> Tensor<Float> {
//        print(matrix)
//        let exp = sqrt(Tensor<Float>(matrix.degree))
        var exp = matrix
        let i = exp.shape[0] - 1
        let j = exp.shape[1] - 1
        let rows = Tensor(stacking: Array((i > 0 ? exp.unstacked()[0...i-1] : []) + (i < exp.shape[0] - 1 ? exp.unstacked()[i+1...exp.shape[0] - 1] : [])))
        let cols = Tensor(stacking: Array((j > 0 ? rows.unstacked(alongAxis: 1)[0...j-1] : []) + (j < rows.shape[1] - 1 ? rows.unstacked(alongAxis: 1)[j+1...rows.shape[1] - 1] : []))).transposed()
        print(rows)
        print(cols)
        let tunedDegreeMatrix = 1 / sqrt(Tensor<Float>(matrix.degree.diagonalPart()))
        let tunedMatrix = tunedDegreeMatrix * Tensor<Float>(matrix) * tunedDegreeMatrix
        let output = matmul(tunedMatrix, entityEmbeddings.embeddings)
        return softmax(matmul(output, outputLayer).flattened())
    }
}
