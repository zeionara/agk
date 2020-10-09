import Foundation
import TensorFlow


public protocol Trainer {
    var nEpochs: Int { get }
    var batchSize: Int { get }
    init(nEpochs: Int, batchSize: Int)
}
