import Foundation
import TensorFlow

public protocol ConvolutionGraphModel: GraphModel {
    var entityEmbeddings: Embedding<Float> { get set }
}
