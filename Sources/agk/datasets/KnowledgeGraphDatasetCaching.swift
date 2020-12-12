import Foundation
import TensorFlow
import Checkpoints
import Logging

var tunedDegreeMatrices = [String: Tensor<Float>]()

let EPSILON: Float = 0.001


extension KnowledgeGraphDataset {
    
    private func computeTunedMatrixInverse(tensorPath: KeyPath<KnowledgeGraphDataset<SourceElement, NormalizedElement>, Tensor<Int8>>) -> Tensor<Float> {
        let matrix = self[keyPath: tensorPath]
        let tunedDegreeMatrix = sqrt(Tensor<Float>(matrix.degree)).inverse
        return matmul(matmul(tunedDegreeMatrix, Tensor<Float>(matrix)), tunedDegreeMatrix)
    }

    private func getCachedTensor(tensorName: String, computeTensor: () -> Tensor<Float>) -> Tensor<Float> {

        let key = "\(name)-\(tensorName)"

        // 0. Try to read from an internal cache

        if tunedDegreeMatrices[key] == Optional.none {

            // 1. Try to read from an external cache

            do {
                let checkpointOperator = try CheckpointReader(
                    checkpointLocation: cachePath,
                    modelName: name
                )
                if checkpointOperator.containsTensor(named: tensorName){
                    let matrix = Tensor<Float>(checkpointOperator.loadTensor(named: tensorName))
                    tunedDegreeMatrices[key] = matrix
                    return matrix
                }
            } catch let exception {
                print(exception)
            }

            // 2. Re-compute and write to the internal and external cache

            var logger = Logger(label: "dataset")
            logger[metadataKey: "name"] = "\(name)"
            logger.logLevel = verbosity
            logger.debug("Cannot read \(tensorName) from cache for dataset \(name). Recomputing...")
            
            let recomputingStartTimestamp = DispatchTime.now().uptimeNanoseconds

            let tunedMatrix = computeTensor()

            logger.debug("Recomputed tensor \(tensorName) for dataset \(name) in \((DispatchTime.now().uptimeNanoseconds - recomputingStartTimestamp) / 1_000_000_000) seconds")

            tunedDegreeMatrices[key] = tunedMatrix

            do {
                var tensors = [tensorName: tunedMatrix]
                
                // Read previosly computed tensors to not to overwrite them

                do {
                    let checkpointReader = try CheckpointReader(
                        checkpointLocation: cachePath,
                        modelName: name
                    )
                    for cachedTensorName in checkpointReader.tensorNames {
                        tensors[cachedTensorName] = Tensor<Float>(checkpointReader.loadTensor(named: cachedTensorName))
                    }
                } catch let exception as NSError {
                    logger.warning("Exisiting tensors for dataset \(name) were not read due to exception: \(exception.debugDescription)")
                }

                // Export results
                
                try CheckpointWriter(
                    tensors: tensors
                ).write(
                    to: getTensorsDatasetCacheRoot(),
                    name: name
                )
            } catch let exception {
                print(exception)
            }

            return tunedMatrix

        } else {
            return tunedDegreeMatrices[key]!
        }
    }
    
    public var tunedAdjecencyMatrixInverse: Tensor<Float> {
        return getCachedTensor(tensorName: "tuned-adjaecency-matrix-inverse") {
            computeTunedMatrixInverse(tensorPath: \.normalizedFrame.adjacencyTensor)
        }   // TODO: Fix tensor name
    }

    public var tunedAdjacencyPairsMatrixInverse: Tensor<Float> {
        return getCachedTensor(tensorName: "tuned-adjacency-pairs-matrix-inverse") {
            computeTunedMatrixInverse(tensorPath: \.normalizedFrame.adjacencyPairsTensor)
        }   // TODO: Fix tensor name
    }

    public var adjacencyPairsTensor: Tensor<Int8> {
        return Tensor<Int8>(
            getCachedTensor(tensorName: "adjacency-pairs-tensor") {
                Tensor<Float>(
                    frame.adjacencyPairsTensor
                )
            }
        )   // TODO: Fix tensor name
    }

    var cachePath: URL {
        return getTensorsDatasetCacheRoot().appendingPathComponent(name)
    }
}
