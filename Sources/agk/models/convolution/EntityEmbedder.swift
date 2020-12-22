import Foundation
import TensorFlow

public protocol EntityEmbedder: GraphModel {
    var entityEmbeddings: Embedding<Float> { get set }
}
