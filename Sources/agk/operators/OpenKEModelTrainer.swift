import Foundation
import TensorFlow
import PythonKit


public struct OpenKEModelTrainer: Trainer {
    public let nEpochs: Int
    public let batchSize: Int // TODO: is not used
    public let openke: PythonObject

    public init(nEpochs: Int, batchSize: Int) {
        self.nEpochs = nEpochs
        self.batchSize = batchSize
        self.openke = Python.import("openke.api")
    }

    public func trainOnGpu<Element>(model: String, frame: TripleFrame<Int32>, dataset: KnowledgeGraphDataset<Element, Int32>) -> PythonObject where Element: PythonConvertible {
        openke.train(
                n_epochs: nEpochs,
                model: model,
                triples: frame.data,
                entity_to_id: dataset.entityId2Index,
                relation_to_id: dataset.relationshipId2Index,
                gpu: true
        )
    }

    public func trainOnCpu<Element>(model: String, frame: TripleFrame<Int32>, dataset: KnowledgeGraphDataset<Element, Int32>) -> PythonObject where Element: PythonConvertible {
        openke.train(
                n_epochs: nEpochs,
                model: model,
                triples: frame.data,
                entity_to_id: dataset.entityId2Index,
                relation_to_id: dataset.relationshipId2Index
        )
    }

    public func train<Element>(model: String, frame: TripleFrame<Int32>, dataset: KnowledgeGraphDataset<Element, Int32>, on device: Device = Device.default) -> PythonObject where Element: PythonConvertible {
//        if (device == Device.defaultXLA) { // TODO: this does not work
//            return trainOnGpu(model: model, frame: frame, dataset: dataset)
//        }
        trainOnCpu(model: model, frame: frame, dataset: dataset)
    }
}
