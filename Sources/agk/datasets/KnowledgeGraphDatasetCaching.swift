import Foundation
import TensorFlow
import Checkpoints

var tunedDegreeMatrices = [String: Tensor<Float>]()

let TUNED_ADJACENCY_MATRIX_INVERSE_TENSOR_KEY = "tuned-adjaecency-matrix-inverse"
let EPSILON: Float = 0.001

extension KnowledgeGraphDataset {
    var tunedAdjecencyMatrixInverse: Tensor<Float> {

        // 0. Try to read from an internal cache

        if tunedDegreeMatrices[name] == Optional.none {
            print("---")
            // 1. Try to read from an external cache

            do {
                let checkpointOperator = try CheckpointReader(
                    checkpointLocation: cachePath,
                    modelName: name
                )
                if checkpointOperator.containsTensor(named: TUNED_ADJACENCY_MATRIX_INVERSE_TENSOR_KEY){
                    let matrix = Tensor<Float>(checkpointOperator.loadTensor(named: TUNED_ADJACENCY_MATRIX_INVERSE_TENSOR_KEY))
                    tunedDegreeMatrices[name] = matrix
                    return matrix
                }
            } catch let exception {
                print(exception)
            }

            // 2. Re-compute and write to the internal and external cache
            let tunedDegreeMatrix = sqrt(Tensor<Float>(normalizedFrame.adjacencyTensor.degree)).inverse
            let tunedMatrix = matmul(matmul(tunedDegreeMatrix, Tensor<Float>(normalizedFrame.adjacencyTensor)), tunedDegreeMatrix) // normalizeWithL2(tensor: matmul(matmul(tunedDegreeMatrix, Tensor<Float>(normalizedFrame.adjacencyTensor)), tunedDegreeMatrix) + EPSILON)

            tunedDegreeMatrices[name] = tunedMatrix

            print("Cannot read \(TUNED_ADJACENCY_MATRIX_INVERSE_TENSOR_KEY) from cache for dataset \(name). Recomputing...")

            do {
                try CheckpointWriter(
                    tensors: [
                            TUNED_ADJACENCY_MATRIX_INVERSE_TENSOR_KEY: tunedMatrix
                        ]
                    ).write(
                        to: getTensorsDatasetCacheRoot(),
                        name: name
                    )
            } catch let exception {
                print(exception)
            }

            return tunedMatrix

        } else {
            return tunedDegreeMatrices[name]!
        }
    }

    var cachePath: URL {
        return getTensorsDatasetCacheRoot().appendingPathComponent(name)
    }
}
