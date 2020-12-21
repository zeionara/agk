import Foundation
import TensorFlow
import Logging

public typealias CVSplit<Element> = (train: TripleFrame<Element>, test: TripleFrame<Element>) where Element: Hashable
public typealias LabelCVSplit<Element> = (train: LabelFrame<Element>, test: LabelFrame<Element>) where Element: Hashable

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

public protocol DataFrame where Element: Hashable{
    associatedtype Element
    func cv(nFolds: Int) -> [(train: Self, test: Self)]
}

public struct LabelFrame<Element>: DataFrame where Element: Hashable {
    let data: [[Element]]
    let device: Device

    public init(data: [[Element]], device: Device) {
        self.data = data
        self.device = device
    }

    public func batched(size: Int, shouldShuffle: Bool = true) -> [LabelFrame] {
        assert(size > 0)

        func addBatch() {
            batches.append(LabelFrame(data: batchSamples, device: device))
            i = 0
            batchSamples = []
        }

        var batches: [LabelFrame] = []
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

    public func split(nChunks: Int, shouldShuffle: Bool = true) -> [LabelFrame] {
        assert(nChunks > 0)
        return batched(size: Int((Float(data.count) / Float(nChunks)).rounded(.up)), shouldShuffle: shouldShuffle)
    }

    public func cv(nFolds: Int) -> [LabelCVSplit<Element>] {
        let enumeratedSplits = split(nChunks: nFolds).enumerated()
        return enumeratedSplits.map { (i: Int, testSplit: LabelFrame) in
            LabelCVSplit<Element>(
                    train: LabelFrame(
                            data: enumeratedSplits.filter { (j: Int, trainPartialSplit: LabelFrame) in
                                j != i
                            }.map { (j: Int, trainPartialSplit: LabelFrame) in
                                trainPartialSplit.data
                            }.reduce([], +),
                            device: device
                    ),
                    test: testSplit
            )
        }
    }
}

public struct TripleFrame<Element>: DataFrame where Element: Hashable {
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
            while true {
                let negativeSample = negativeFrame.generateSample(positiveSample: positiveSample, corruptionDegree: corruptionDegree)!
                corruptedTriples.append(negativeSample)
                i += 1
                if i >= n {
                    break
                }
            }
            return corruptedTriples
        }
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
    }

    public func next() -> [Element]? {
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
    public let normalizedNegativeFrame: NegativeFrame<NormalizedElement>
    public let entityId2Index: [SourceElement: NormalizedElement]
    public let entityId2Text: [NormalizedElement: String]
    public let entityIndex2Id: [NormalizedElement: SourceElement]
    public let relationshipId2Index: [SourceElement: NormalizedElement]
    public let relationshipIndex2Id: [NormalizedElement: SourceElement]
    public let device: Device
    public let path: String
    public let name: String
    public let verbosity: Logger.Level

    static func readStringPairs(path: String) throws -> [(String, String)] {
        let dir = URL(fileURLWithPath: #file.replacingOccurrences(of: "Sources/agk/datasets/KnowledgeGraphDataset.swift", with: ""))
        let fileContents = try String(
            contentsOf: dir.appendingPathComponent("data").appendingPathComponent(path),
            encoding: .utf8
        )
        let data: [(String, String)] = fileContents.split(separator: "\n").map {
            let stringTuple = String($0).components(separatedBy: "\t")
            return (stringTuple[0], stringTuple[1])
        }
        return data
    }

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

//
//    These methods should be defined in the extensions using specific source and normalized element types
//
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
            path: String, classes: String? = Optional.none, texts: String? = Optional.none, device: Device = Device.default,
            intToNormalizedElement: (Int) -> NormalizedElement, stringToNormalizedElement: (String) -> NormalizedElement, stringToSourceElement: (String) -> SourceElement,
            sourceToNormalizedElement: (SourceElement) -> NormalizedElement,
            verbosity: Logger.Level = .debug
    ) {
        self.path = path
        self.verbosity = verbosity
        self.name = path.components(separatedBy: "/").last!.components(separatedBy: ".").first!
        let frame_ = TripleFrame(data: try! KnowledgeGraphDataset.readData(path: path, stringToSourceElement: stringToSourceElement), device: device)
        let negativeFrame_ = NegativeFrame(frame: frame_)
        let entityNormalizationMappings = makeNormalizationMappings(source: frame_.entities, destination: Array(0...frame_.entities.count - 1).map {
            intToNormalizedElement($0)
        })
        let relationshipNormalizationMappings = makeNormalizationMappings(source: frame_.relationships, destination: Array(0...frame_.entities.count - 1).map {
            intToNormalizedElement($0)
        })

        if let classes_ = classes {
            labelFrame = LabelFrame(
                    data: (try! KnowledgeGraphDataset.readData(path: classes_) { s in
                        stringToSourceElement(s)
                    }).map { row in
                        return [entityNormalizationMappings.forward[row.first!]!, sourceToNormalizedElement(row.last!)]
                    }.sorted {
                        $0.first! < $1.first!
                    },
                    device: device
            )
        } else {
            labelFrame = Optional.none
        }

        var id2Text = [NormalizedElement: String]()
        if let texts_ = texts {
            try! KnowledgeGraphDataset.readStringPairs(path: texts_).map{ row in
                id2Text[entityNormalizationMappings.forward[stringToSourceElement(row.0)]!] = row.1
            }
        }

        let normalizedFrame_ = normalize(frame_, entityNormalizationMappings.forward, relationshipNormalizationMappings.forward)

        self.device = device

        frame = frame_
        negativeFrame = negativeFrame_
        normalizedFrame = normalizedFrame_
        normalizedNegativeFrame = NegativeFrame(frame: normalizedFrame)

        entityId2Index = entityNormalizationMappings.forward
        entityIndex2Id = entityNormalizationMappings.backward
        relationshipId2Index = relationshipNormalizationMappings.forward
        relationshipIndex2Id = relationshipNormalizationMappings.backward
        entityId2Text = id2Text
    }

    public func getAdjacencyPairsIndices(labels: LabelFrame<Int32>) -> Tensor<Int32> {
        let nEntities = Int32(frame.entities.count)
        let nRelationships = Int32(frame.relationships.count)
        let offset = nEntities * nRelationships
        
        func getIndex(entityIndex: Int32, relationshipIndex: Int32) -> Int32 {
            return nEntities * relationshipIndex + offset + entityIndex
        }

        func getTensor(_ entityIndex: Int32) -> Tensor<Int32> {
            return Tensor(
                (0..<nRelationships).map { relationshipIndex in
                    Tensor<Int32>(
                        [
                            Int32(nEntities * relationshipIndex) + entityIndex,
                            getIndex(entityIndex: entityIndex, relationshipIndex: relationshipIndex)
                        ],
                        on: device
                    )
                }
            ).flattened()
        }
        
        return Tensor(
            labels.indices.unstacked().map { entityIndex in
                getTensor(entityIndex.scalar!)
            }
        )
    }
}