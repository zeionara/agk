import Foundation
import TensorFlow

public typealias CVSplit<Element> = (train: TripleFrame<Element>, test: TripleFrame<Element>) where Element: Hashable

public enum CorruptionDegree: Int {
    case none = 3
    case eitherHeadEitherTail = 2
    case headAndTail = 1
    case complete = 0
}

func getRandomNumbers(maxNumber: Int, listSize: Int) -> Set<Int> {
    var randomNumbers = Set<Int>()
    while randomNumbers.count < listSize && randomNumbers.count - 1 < maxNumber {
        let randomNumber = Int.random(in: 0...maxNumber)
        randomNumbers.insert(randomNumber)
    }
    return randomNumbers
}

public struct LabelFrame<Element> {
    let data: [[Element]]
    let device: Device

    public init(data: [[Element]], device: Device) {
        self.data = data
        self.device = device
    }
}

public struct TripleFrame<Element> where Element: Hashable {
    public let data: [[Element]]
    let device: Device
    var entities_: [Element]?
    var relationships_: [Element]?

    public init(data: [[Element]], device: Device, entities_: [Element]? = Optional.none, relationships_: [Element]? = Optional.none) {
        self.data = data
        self.device = device
        self.entities_ = entities_
        self.relationships_ = relationships_
    }

    public func cv(nFolds: Int) -> [CVSplit<Element>] {
        let enumeratedSplits = split(nChunks: nFolds).enumerated()
        return enumeratedSplits.map { (i: Int, testSplit: TripleFrame) in
            CVSplit<Element>(
                    train: TripleFrame(
                            data: enumeratedSplits.filter { (j: Int, trainPartialSplit: TripleFrame) in
                                j != i
                            }.map { (j: Int, trainPartialSplit: TripleFrame) in
                                trainPartialSplit.data
                            }.reduce([], +),
                            device: device,
                            entities_: entities,
                            relationships_: relationships
                    ),
                    test: testSplit
            )
        }
    }

    public func batched(sizes: [Int], shouldShuffle: Bool = true) -> [TripleFrame] {
        func addBatch() {
            batches.append(TripleFrame(data: batchSamples, device: device, entities_: entities, relationships_: relationships))
            i = 0
            batchSamples = []
        }

        var batches: [TripleFrame] = []
        var batchSamples: [[Element]] = []
        var i = 0
        var currentSizeId = 0
        for sample in shouldShuffle ? data.shuffled() : data {
            if ((i == 0 && sizes[currentSizeId] == 0) || (i > 0 && i % sizes[currentSizeId] == 0)) {
                addBatch()
                if currentSizeId < data.count - 1 {
                    currentSizeId += 1
                    i = 0
                } else {
                    break
                }
            }
            batchSamples.append(sample)
            i += 1
        }
        if !batchSamples.isEmpty {
            addBatch()
        }
        return batches
    }

    public func getCombinations(k: Int) -> [TripleFrame] {
        self.data.getCombinations(k: k, n: self.data.count).map { combination in
            TripleFrame(data: combination, device: device, entities_: entities, relationships_: relationships)
        }
    }

    public func batched(size: Int, shouldShuffle: Bool = true) -> [TripleFrame] {
        assert(size > 0)

        func addBatch() {
            batches.append(TripleFrame(data: batchSamples, device: device, entities_: entities, relationships_: relationships))
            i = 0
            batchSamples = []
        }

        var batches: [TripleFrame] = []
        var batchSamples: [[Element]] = []
        var i = 0
        for sample in shouldShuffle ? data.shuffled() : data {
            if (i % size == 0 && i > 0) {
                addBatch()
            }
            batchSamples.append(sample)
            i += 1
        }
        if !batchSamples.isEmpty {
            addBatch()
        }
        return batches
    }

    public func split(nChunks: Int, shouldShuffle: Bool = true) -> [TripleFrame] {
        assert(nChunks > 0)
        return batched(size: Int((Float(data.count) / Float(nChunks)).rounded(.up)), shouldShuffle: shouldShuffle)
    }

    public func split(proportions: [Float], shouldShuffle: Bool = true) -> [TripleFrame] {
        assert(proportions.reduce(0.0, +) == 1.0)
        return batched(
                sizes: proportions.map { ratio in
                    Int((ratio * Float(data.count)).rounded())
                },
                shouldShuffle: shouldShuffle
        )
    }

    public var entities: [Element] {
        if let entities__ = entities_ {
            return entities__
        }
        return (data[column: 0] + data[column: 1]).unique()
    }

    public var relationships: [Element] {
        if let relationships__ = relationships_ {
            return relationships__
        }
        return data[column: 2].unique()
    }

    public func sampleNegativeFrame(negativeFrame: NegativeFrame<Element>, n: Int = 1, corruptionDegree: CorruptionDegree = CorruptionDegree.eitherHeadEitherTail) -> TripleFrame<Element> {
        var j = 0
        let start_timestamp = DispatchTime.now().uptimeNanoseconds
        var negativeSamples: [[[Element]]] = data.map { positiveSample in
            j += 1
            var i = 0
            var corruptedTriples = [[Element]]()
//            while i < n {
            // for negativeSample in negativeFrame.generateData(positiveSample: positiveSample, corruptionDegree: corruptionDegree) {
            while true {
                let negativeSample = negativeFrame.generateSample(positiveSample: positiveSample, corruptionDegree: corruptionDegree)!
//                if (
//                           negativeSample[2] == positiveSample[2] && (
//                                   (
//                                           (negativeSample[0] != positiveSample[0] && negativeSample[1] != positiveSample[1]) && corruptionDegree == CorruptionDegree.headAndTail
//                                   ) || (
//                                           (
//                                                   (negativeSample[0] == positiveSample[0] && negativeSample[1] != positiveSample[1]) ||
//                                                           (negativeSample[0] != positiveSample[0] && negativeSample[1] == positiveSample[1])
//                                           ) &&
//                                                   corruptionDegree == CorruptionDegree.eitherHeadEitherTail
//                                   )
//                           )
//                   ) || (
//                        (negativeSample[0] != positiveSample[0] && negativeSample[1] != positiveSample[1] && negativeSample[2] != positiveSample[2]) && corruptionDegree == CorruptionDegree.complete
//                ) {
                corruptedTriples.append(negativeSample)
                i += 1
                if i >= n {
                    break
                }
//                }
            }
//            }
            return corruptedTriples
        }
        // print("Generated negative frame in \((DispatchTime.now().uptimeNanoseconds - start_timestamp) / 1_000_000_000) seconds")
        return TripleFrame<Element>(data: negativeSamples.reduce([], +), device: device, entities_: entities, relationships_: relationships)
    }

    public var negative: NegativeFrame<Element> {
        NegativeFrame<Element>(frame: self)
    }

    public func sample(size: Int) -> TripleFrame {
        TripleFrame(data: getRandomNumbers(maxNumber: data.count - 1, listSize: size).map {
            data[$0]
        }, device: device, entities_: entities, relationships_: relationships)
    }
}

public func makeNormalizationMappings<KeyType, ValueType>(source: [KeyType], destination: [ValueType]) -> (forward: Dictionary<KeyType, ValueType>, backward: Dictionary<ValueType, KeyType>) {
    (
            Dictionary(
                    uniqueKeysWithValues: zip(
                            source.map {
                                $0
                            }, destination
                    )
            ),
            Dictionary(
                    uniqueKeysWithValues: zip(
                            destination.map {
                                $0
                            }, source
                    )
            )
    )
}

private func addTriple<Element>(triples: inout [Element: [Element: [Element: Bool]]], triple: [Element]) {
    let head = triple[0]
    let tail = triple[1]
    let relationship = triple[2]
    if triples[head] == nil {
        triples[head] = [Element: [Element: Bool]]()
        triples[head]![tail] = [Element: Bool]()
        triples[head]![tail]![relationship] = true
    } else {
        if triples[head]![tail] == nil {
            triples[head]![tail] = [Element: Bool]()
            triples[head]![tail]![relationship] = true
        } else {
            if triples[head]![tail]![relationship] == nil {
                triples[head]![tail]![relationship] = true
            }
        }
    }
}

public class NegativeSampleGenerator<Element>: IteratorProtocol, Sequence where Element: Hashable {
    let positiveTriples: [Element: [Element: [Element: Bool]]]
    let entities: [Element]
    let relationships: [Element]
//    let positiveSample: [Element]?
//    let corruptionDegree: CorruptionDegree?
    var history: [Element: [Element: [Element: Bool]]]

    public init(frame: TripleFrame<Element>, history: [Element: [Element: [Element: Bool]]]? = nil) {
        var positiveTriples_ = [Element: [Element: [Element: Bool]]]()
        for positiveTriple in frame.data {
            addTriple(triples: &positiveTriples_, triple: positiveTriple)
        }
        entities = frame.entities
        relationships = frame.relationships
        self.history = history ?? [Element: [Element: [Element: Bool]]]()
        positiveTriples = positiveTriples_
//        self.positiveSample = positiveSample
//        self.corruptionDegree = corruptionDegree
    }

    public func next() -> [Element]? {
//        if let positiveSample = positiveSample, let corruptionDegree = corruptionDegree {
        // var secondIteration = false
//        } else {
        while true {
            let head = entities.randomElement()!
            let tail = entities.randomElement()!
            let relationship = relationships.randomElement()!
            let triple = [head, tail, relationship]
            if (positiveTriples[head]?[tail]?[relationship] == nil && history[head]?[tail]?[relationship] == nil) {
                addTriple(triples: &history, triple: triple)
                return triple
            }
        }
//        }
    }

    public func next(positiveSample: [Element], corruptionDegree: CorruptionDegree) -> [Element]? {
        while true {
            let relationship = corruptionDegree == .complete ? relationships.randomElement()! : positiveSample[2]
            var head: Element = positiveSample[0]
            var tail: Element = positiveSample[1]
            if (corruptionDegree == .complete || corruptionDegree == .headAndTail) {
                head = entities.randomElement()!
                tail = entities.randomElement()!
            } else if corruptionDegree == .eitherHeadEitherTail {
                let shouldCorruptHead = Int.random(in: 0...1)
                if shouldCorruptHead == 1 {
                    head = entities.randomElement()!
                    tail = positiveSample[1]
                } else {
                    tail = entities.randomElement()!
                    head = positiveSample[0]
                }
            }
            let triple = [head, tail, relationship]
            if (positiveTriples[head]?[tail]?[relationship] == nil && history[head]?[tail]?[relationship] == nil) {
                addTriple(triples: &history, triple: triple)
                return triple
            }
        }
    }

    public func resetHistory() {
        history = [Element: [Element: [Element: Bool]]]()
    }
}

public struct NegativeFrame<Element> where Element: Hashable {
    public let data: NegativeSampleGenerator<Element>
    public let frame: TripleFrame<Element>

    public init(frame: TripleFrame<Element>) {
        self.frame = frame
        data = NegativeSampleGenerator(frame: frame)
    }

    public func generateSample(positiveSample: [Element], corruptionDegree: CorruptionDegree) -> [Element]? {
        data.next(positiveSample: positiveSample, corruptionDegree: corruptionDegree)
    }

    public func batched(sizes: [Int], device: Device = Device.default) -> [TripleFrame<Element>] {
        func addBatch() {
            batches.append(TripleFrame(data: batchSamples, device: device, entities_: data.entities, relationships_: data.relationships))
            i = 0
            batchSamples = []
        }

        var batches: [TripleFrame<Element>] = []
        var batchSamples: [[Element]] = []
        var i = 0
        var currentSizeId = 0
        for sample in data {
            if ((i == 0 && sizes[currentSizeId] == 0) || (i > 0 && i % sizes[currentSizeId] == 0)) {
                addBatch()
                currentSizeId += 1
                i = 0
                if currentSizeId >= sizes.count {
                    break
                }
            }
            batchSamples.append(sample)
            i += 1
        }
        if !batchSamples.isEmpty {
            addBatch()
        }
        return batches
    }

    public func batched(size: Int, nBatches: Int, device: Device = Device.default) -> [TripleFrame<Element>] {
        assert(size > 0)

        func addBatch() {
            batches.append(TripleFrame<Element>(data: batchSamples, device: device, entities_: data.entities, relationships_: data.relationships))
            i = 0
            batchSamples = []
        }

        var batches: [TripleFrame<Element>] = []
        var batchSamples: [[Element]] = []
        var i = 0
        var j = 0
        for sample in data {
            if (i % size == 0 && i > 0) {
                addBatch()
                j += 1
            }
            batchSamples.append(sample)
            i += 1
            if (j % nBatches == 0 && j > 0) {
                break
            }
        }
        if !batchSamples.isEmpty {
            addBatch()
        }
        return batches
    }
}

//public func makeNegativeFrame<Element>(frame: TripleFrame<Element>, size: Int? = Optional.none) -> TripleFrame<Element> {
//    var negativeSamples: [[Element]] = []
//    var positiveMap = [Element: [Element: [Element: Bool]]]()
//
//    for item in frame.data {
//        let head = item[0]
//        let tail = item[1]
//        let relationship = item[2]
//        if triples[head] == nil {
//            triples[head] = [Element: [Element: Bool]]()
//            triples[head]![tail] = [Element: Bool]()
//            triples[head]![tail]![relationship] = true
//        } else {
//            if triples[head]![tail] == nil {
//                triples[head]![tail] = [Element: Bool]()
//                triples[head]![tail]![relationship] = true
//            } else {
//                if triples[head]![tail]![relationship] == nil {
//                    triples[head]![tail]![relationship] = true
//                }
//            }
//        }
//    }
////    print(triples)
////    let samples = frame.entities.map { head in
////        frame.entities.map{ tail in
////            frame.relationships.map{ (relationship) -> [Element] in
////                [head, tail, relationship]
////            }
////        }.reduce([], +)
////    }.reduce([], +).filter{ item in
////        !data.contains(item)
////    }
//
//    let unwrappedSize = size ?? frame.data.count * 10
//    var triple_counter = 0
//    var start = DispatchTime.now()
//    for (i, head) in frame.entities.enumerated() {
////        print("Handled \(i) / \(frame.entities.count) heads")
//        for (j, tail) in frame.entities.enumerated() {
////            print("Handled \(j) / \(frame.entities.count) tails")
//            for (k, relationship) in frame.relationships.enumerated() {
////                print("Handled \(k) / \(frame.relationships.count) relationships")
//                if (triple_counter < unwrappedSize) {
//                    start = DispatchTime.now()
//                    if triples[head]?[tail]?[relationship] == nil {
//                        negativeSamples.append([head, tail, relationship])
//                        triple_counter += 1
//                        print(triple_counter)
//                    }
//                } else {
//                    print("Handled in \((DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000) ms")
//                    return TripleFrame(data: negativeSamples, device: frame.device, entities_: frame.entities, relationships_: frame.relationships)
//                }
//            }
//        }
//    }
//    print("Handled in \((DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000) ms")
//    return TripleFrame(data: negativeSamples, device: frame.device, entities_: frame.entities, relationships_: frame.relationships)
//}


private func normalize<SourceElement, NormalizedElement>(
        _ frame: TripleFrame<SourceElement>,
        _ entityNormalizationMapping: [SourceElement: NormalizedElement],
        _ relationshipNormalizationMapping: [SourceElement: NormalizedElement]
) -> TripleFrame<NormalizedElement> {
    TripleFrame(
            data: frame.data.map {
                [
                    entityNormalizationMapping[$0[0]]!,
                    entityNormalizationMapping[$0[1]]!,
                    relationshipNormalizationMapping[$0[2]]!
                ]
            },
            device: frame.device
    )
}

public struct KnowledgeGraphDataset<SourceElement, NormalizedElement> where SourceElement: Hashable, NormalizedElement: Hashable, NormalizedElement: Comparable {
    public let frame: TripleFrame<SourceElement>
    public let labelFrame: LabelFrame<NormalizedElement>?
    public let normalizedFrame: TripleFrame<NormalizedElement>
    public let negativeFrame: NegativeFrame<SourceElement>
//    public let sampledNegativeFrame: TripleFrame
    public let normalizedNegativeFrame: NegativeFrame<NormalizedElement>
//    public let normalizedSampledNegativeFrame: TripleFrame
    public let entityId2Index: [SourceElement: NormalizedElement]
    public let entityIndex2Id: [NormalizedElement: SourceElement]
    public let relationshipId2Index: [SourceElement: NormalizedElement]
    public let relationshipIndex2Id: [NormalizedElement: SourceElement]
    public let device: Device
    public let path: String

    static func readData<Element>(path: String, stringToSourceElement: (String) -> Element) throws -> [[Element]] {
        let dir = URL(fileURLWithPath: #file.replacingOccurrences(of: "Sources/agk/datasets/KnowledgeGraphDataset.swift", with: ""))
        let fileContents = try String(
                contentsOf: dir.appendingPathComponent("data").appendingPathComponent(path),
                encoding: .utf8
        )
        let data: [[Element]] = fileContents.split(separator: "\n").map {
            String($0).split(separator: "\t").compactMap {
                stringToSourceElement(String($0))
            }
        }
        return data
    }

//    public static func stringToSourceElement(_ string: String) -> SourceElement? {
//        print(string)
//        return Optional.none
//    }
//
//    public static func intToNormalizedElement(_ item: Int) -> NormalizedElement? {
//        print(item)
//        return Optional.none
//    }

    public init(
            path: String, classes: String? = Optional.none, device: Device = Device.default,
            intToNormalizedElement: (Int) -> NormalizedElement, stringToNormalizedElement: (String) -> NormalizedElement, stringToSourceElement: (String) -> SourceElement,
            sourceToNormalizedElement: (SourceElement) -> NormalizedElement
    ) {
        self.path = path
//        print("Loading frame...")
        let frame_ = TripleFrame(data: try! KnowledgeGraphDataset.readData(path: path, stringToSourceElement: stringToSourceElement), device: device)
//        print("Generating negative frame...")
        let negativeFrame_ = NegativeFrame(frame: frame_)
//        let sampledNegativeFrame_ = makeSampledNegativeFrame(frame: frame_, negativeFrame: negativeFrame_)

//        print("Building entity normalization mappings...")
        let entityNormalizationMappings = makeNormalizationMappings(source: frame_.entities, destination: Array(0...frame_.entities.count - 1).map {
            intToNormalizedElement($0)
        })
//        print("Setting up relationship normalization mappings...")
        let relationshipNormalizationMappings = makeNormalizationMappings(source: frame_.relationships, destination: Array(0...frame_.entities.count - 1).map {
            intToNormalizedElement($0)
        })

        if let classes_ = classes {
            labelFrame = LabelFrame(
                    data: (try! KnowledgeGraphDataset.readData(path: classes_) { s in
                        stringToSourceElement(s)
                    }).map { row in
                        [entityNormalizationMappings.forward[row.first!]!, sourceToNormalizedElement(row.last!)]
                    }.sorted {
                        $0.first! < $1.first!
                    },
                    device: device
            )
        } else {
            labelFrame = Optional.none
        }
        let normalizedFrame_ = normalize(frame_, entityNormalizationMappings.forward, relationshipNormalizationMappings.forward)

        self.device = device

        frame = frame_
        negativeFrame = negativeFrame_
//        sampledNegativeFrame = sampledNegativeFrame_
        normalizedFrame = normalizedFrame_
        normalizedNegativeFrame = NegativeFrame(frame: normalizedFrame)
//        normalizedSampledNegativeFrame = normalize(sampledNegativeFrame_, entityNormalizationMappings.forward, relationshipNormalizationMappings.forward)

        entityId2Index = entityNormalizationMappings.forward
        entityIndex2Id = entityNormalizationMappings.backward
        relationshipId2Index = relationshipNormalizationMappings.forward
        relationshipIndex2Id = relationshipNormalizationMappings.backward
    }
}