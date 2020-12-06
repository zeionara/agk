# Agk Tool (being reversed stands for "Knowledge Graphs Applications")
A library for building recommendation systems.
## Example of usage
To cross-validate implemented models of knowledge graph representation on a dataset and calculate metrics for comparing multiple solutions (being in the root project directory):
1. Compile the source code
```sh
swift build --product recommenders
```
2. Run tesing process (for the complete list of supported models and cross-validation parameters see file `Sources/recommenders/main.swift`)
```sh
.build/debug/recommenders -m transe -d truncated-dataset-normalized.txt
```
