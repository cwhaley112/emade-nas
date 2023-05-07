import random
names = [#"DEPTH_ESTIMATE",
                 "ARGMAX", "ARGMIN",
                 #"GMM",
                 "KNN", "SVM", "KMEANS",
                 "RAND_FOREST", 
                 "BOOSTING", 
                 "DECISION_TREE",
                 "LOGR", "LINSVC", "SGD",
                 "PASSIVE",
                 "EXTRATREES",
                 "XGBOOST",
                 "LIGHTGBM",
                 "BOOSTING_REGRESSION",
                 "ADABOOST_REGRESSION",
                 "RANDFOREST_REGRESSION",
                 "SVM_REGRESSION",
                 "KNN_REGRESSION"]
params = [#{'sampling_rate':1, 'off_nadir_angle':20.0},
          {'sampling_rate':1}, {'sampling_rate':1},
          #{'n_components':2, 'covariance_type':0},
          {'K': 3, 'weights':0}, {'C':1.0, 'kernel':0}, {'n_clusters':8}, 
          {'n_estimators': 100, 'criterion':0, 'max_depth': 3, 'class_weight':0},
          {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3}, 
          {'criterion':0, 'splitter':0},
          {'penalty':0, 'C':1.0}, {'C':1.0}, {'penalty':0, 'alpha':0.0001},
          {'C':1.0},
          {'n_estimators': 100, 'max_depth':6, 'criterion':0},
          {
            'learning_rate': 0.1,
            'max_depth': 3,
            'n_estimators': 100,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0},
          {
            'max_depth': -1,
            'learning_rate': 0.1,  # shrinkage_rate
            'boosting_type': 0,
            'num_leaves': 31},
          {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3},
          {'learning_rate': 0.1, 'n_estimators': 100},
          {'n_estimators': 100, 'criterion':0},
          {'kernel':0},
          {'K': 3, 'weights':0}
          ]


f = open("rkan3", "w")
def printf(s):
    f.write(str(s))
    f.write("\n")
def applySentiment(s="ARG0"):
    ret = []
    b = ["trueBool", "falseBool"]
    for bb in b:
        ret.append("Sentiment(" + s + ", " + bb + ")")
    return ret
def applyStem(s="ARG0"):
    return ["Stemmatizer(" + s + ", " + str(i) + ", 0)" for i in range(3)]

def applyVector(l):
    ret = []
    w2v = ["Word2VecVectorizer("]
    rest = ["HashingVector(", "TfidfVectorizer(", "CountVectorizer("]
    for ll in l:
        #w2v
        start = w2v[0]
        stop = random.choice(range(5))
        size = random.choice(range(100, 600, 100))
        window = random.choice(range(100, 600, 100))
        minC = random.choice(range(3))
        for b in ["trueBool", "falseBool"]:
            seed = start + ll + ", "
            seed += str(stop) + ", "
            seed += str(size) + ", "
            seed += str(window) + ", "
            seed += str(minC) + ", "
            seed += b + ")"
            ret.append(seed)
        # for start in w2v:
        #   for stop in range(1):
        #       for size in random.choice(range(100, 600, 100)):
        #           for window in random.choice(range(100, 600, 100)):
        #               for minC in range(3):
        #                   for b in ["trueBool", "falseBool"]:
                                
        #everyone else
        for start in rest:
            for b in [", trueBool, ", ", falseBool, "]:
                ngram = "0, 2, "
                i = random.choice(range(5))
                seed = start + ll + b + ngram
                seed += str(i) + ")"
                ret.append(seed)
            # for b in [", trueBool, ", ", falseBool, "]:
            #   for ngram in ["0, 0, ", "0, 1, ", "0, 2, ", "0, 3, "]:
            #       for i in range(5): #stopword list
            #           seed = start + ll + b + ngram
            #           seed += str(i) + ")"
            #           ret.append(seed)
    return ret
def applyLearner(ll):
    ret = []
    for l in ll:
        for name, param in random.sample(list(zip(names, params)), 4):
            seed = "Learner(" + l + ", "
            seed += "learnerType(" + "\'"+name+"\'" + ", " + str(param) + ", \'SINGLE\', None))"
            ret.append(seed)
    return ret
if __name__ == "__main__":
    inputs = applyVector(applyStem() + ["ARG0"]) + applySentiment()
    # for i in inputs:
    #   printf(i)
    print(len(inputs))
    seeds = applyLearner(inputs)
    for i in seeds:
        printf(i)
    f.close()
    print(len(seeds))