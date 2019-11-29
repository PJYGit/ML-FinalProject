from math import log


def creatTree(dataset, features):
    targetFeatures = [feature[-1] for feature in dataset]

    if targetFeatures.count(targetFeatures[0]) == len(targetFeatures):
        return targetFeatures[0]

    best = get_best_feature(dataset)

    best_feature = features[best]
    DTree = {best_feature: {}}
    del (features[best])
    values = [feature[best] for feature in dataset]
    temp = set(values)
    for value in temp:
        partation = get_partation(dataset, best, value)
        part_features = features[:]
        DTree[best_feature][value] = creatTree(partation, part_features)
    return DTree


def get_entropy(dataset):
    num = len(dataset)
    counts = {}
    for feature in dataset:
        cur_Feature = feature[-1]
        if cur_Feature not in counts.keys():
            counts[cur_Feature] = 0
        counts[cur_Feature] += 1
    entropy = 0.0
    for key in counts:
        p = float(counts[key]) / num
        entropy -= p * log(p, 2)
    return entropy


def get_partation(dataset, featureIndex, value):
    partation = []
    for r in dataset:
        if r[featureIndex] == value:
            partation_r = r[:featureIndex]
            partation_r.extend(r[featureIndex + 1:])
            partation.append(partation_r)
    return partation


def get_best_feature(dataset):
    num = len(dataset[0]) - 1
    entropy = get_entropy(dataset)
    IG = 0.0
    best = -1
    for i in range(num):
        featureValues = [feature[i] for feature in dataset]
        temp = set(featureValues)
        partation_entropy = 0.0
        for value in temp:
            partation = get_partation(dataset, i, value)
            prob_i = len(partation) / float(len(dataset))
            partation_entropy += prob_i * get_entropy(partation)
        partation_IG = entropy - partation_entropy
        if (partation_IG > IG):
            IG = partation_IG
            best = i
    return best


def feature_detection(DTree, features, query):
    best_feature = list(DTree.keys())[0]
    subtree = DTree[best_feature]
    Index = features.index(best_feature)
    for split in subtree.keys():
        if query[Index] == split:
            if isinstance(subtree[split], dict):
                target_feature = feature_detection(subtree[split], features, query)
            else:
                target_feature = subtree[split]
    return target_feature


if __name__ == '__main__':
    dataset = [[0, 1, 0, 'true'], [0, 0, 0, 'false'], [0, 1, 0, 'true'],
               [1, 0, 0, 'false'], [1, 0, 1, 'true'], [1, 0, 0, 'false']]
    features = ['Good Behavior', 'Age', 'Drug Dependent']
    temp = features[:]

    decision_tree = creatTree(dataset, features)
    print('The decision tree is:\n', decision_tree)

    print('The prediction of \'Good Behavior = false, Age < 30 = false, Drug Dependence = true\' is:')
    prediction = feature_detection(decision_tree, temp, [0, 0, 1])
    print(prediction)

    print('The prediction of \'Good Behavior = true, Age < 30 = true, Drug Dependence = false\' is:')
    prediction = feature_detection(decision_tree, temp, [1, 1, 0])
    print(prediction)
