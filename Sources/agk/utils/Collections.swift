extension Array where Element: Equatable {
    public func count(_ externalElement: Element) -> Int {
        self.filter { (internalElement: Element) in
            internalElement == externalElement
        }.count
    }
}

extension Array where Element == Float {
    public var mean: Float {
        Float(self.reduce(Element(0), +)) / Float(self.count)
    }
}

extension Array {
    func getCombination(k: Int) -> [Int] {
        var combination = [Int]()
        while combination.count < k {
            let randomElement = Int.random(in: 0..<self.count)
            if !(combination.contains(randomElement)) {
                combination.append(randomElement)
            }
        }
        return combination.sorted()
    }

    func getCombinations(k: Int, n: Int) -> [[Element]] {
        assert(k <= self.count)
        var combinations: [[Int]] = []
        while combinations.count < n {
            let combination = self.getCombination(k: k)
            if !(combinations.contains(combination)) {
                combinations.append(combination)
            }
        }
//        for i in 0..<k {
//            if combinations.count == 0 {
//                for j in 0..<self.count {
//                    combinations.append([j])
//                }
//            } else {
//                var nCombinationsFromPreviousStep = 0
//                for (combinationIndex, combination) in combinations.enumerated() {
//                    print("handled \(combinationIndex) / \(combinations.count) combinations")
//                    for j in 0..<self.count {
//                        if !combination.contains(j) {
//                            let newCombination = (combination + [j]).sorted()
//                            if !combinations.contains(newCombination) {
//                                combinations.append(newCombination)
//                            }
//                        }
//                    }
//                    nCombinationsFromPreviousStep += 1
//                }
//                combinations = [[Int]](combinations.dropFirst(nCombinationsFromPreviousStep))
//            }
//        }
        return combinations.map { combination in
            combination.map {
                self[$0]
            }
        }
//        return combinations
    }

    func toDict<K, V>(map: (Element) -> (key: K, value: V)?) -> [K: V] {
        var dict = [K: V]()
        for element in self {
            if let (key, value) = map(element) {
                dict[key] = value
            }
        }
        return dict
    }
}

public extension Array where Element == (String, String) {
    subscript(index: String) -> String? {
        for item in self {
            if item.0 == index {
                return item.1
            }
        }
        return Optional.none
    } 
}
