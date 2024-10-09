SUPPORTED_CLASSIFIER = ["RF", "XT", "GB", "AB", "BNB", "DTREE", "GNB", "KNN_CLASS", "LDA", "LSVM",
                        "MLP", "MNB", "PA", "QDA", "SGD", "SVM"]

SUPPORTED_COMBINER = ["BC", "BKS", "DTEMP", "DS", "COS", "KNN_COMB", "MAMV", "MIMV", "MLE", "NB",
                      "NN", "AVG", "WV"]

SOLVER = ["RS", "BO"]

# Algorithm Type
CLASSIFICATION = "classification"
COMBINATION = "combination"
PREPROCESSOR = "preprocessor"
PIPELINE = "pipeline"

# Input Data Types
NUMERICAL = "continuous"
CATEGORICAL = "categorical"
MIXED = "mixed"

# Classification Problem
BINARY = "binary"
MULTICLASS = "multiclass"
MULTILABEL = "multilabel"

# Output Types (+ Input Types for Combiners)
LABELS = "labels"  # If classifier Output: A label vector, If combiner In-/Output: A label matrix
CONTINUOUS_OUT = "continuous_out"

# Evaluation Metrics
ACCURACY = "accuracy"
BALANCED_ACCURACY = "balanced_accuracy"
PRECISION = "precision"
PRECISION_MICRO = "precision_micro"
PRECISION_MACRO = "precision_macro"
RECALL = "recall"
RECALL_MICRO = "recall_micro"
RECALL_MACRO = "recall_macro"
F1_MICRO = "f1_micro"
F1_MACRO = "f1_macro"
JACCARD_MICRO = "jaccard_micro"
JACCARD_MACRO = "jaccard_macro"
AP_MICRO = "apmicro"
AP_MACRO = "apmacro"
MC = "meanconfidence"
ROC_AUC_OVR = "roc_auc_ovr"
ROC_AUC_OVO = "roc_auc_ovo"

# Diversity Metrics
YULES_Q = "yulesq"
YULES_Q_NORM = "yulesq_norm"
CORRELATION_COEFFICIENT = "correlation"
CORRELATION_COEFFICIENT_NORM = "correlation_norm"
DISAGREEMENT = "disagreement"
DOUBLEFAULT = "doublefault"
DOUBLEFAULT_NORM = "doublefault_norm"
KAPPA = "kappa-error"
MC_DISAGREEMENT = "mc_disagreement"
MC_DOUBLEFAULT = "mc_doublefault"
MC_DOUBLEFAULT_NORM = "mc_doublefault_norm"

# Implicit Diversity Methods
BOOTSTRAP = "BS"
RSM = "RSM"
PS = "PS"
RP = "RP"  # Random Patches with bootstrapping and rsm
RPPS = "RPPS"  # Random patches with pasting and rsm
NOISE = "NOISE"
CV = "CV"
FLIP = "FLIP"

# Preprocessing methods
STANDARD = "STD"
MINMAX = "MINMAX"
NORMALIZATION = "NORM"
ROBUST = "ROBUST"
POWERTRF = "POWERTRF"
QUANTILETRF = "QUANTILETRF"
ENCODER = "ENC"

# Feature Engineering methods
FASTICA = "ICA"
PCATR = "PCA"
NYSTROEM = "NYSTROEM"
KITCHEN = "RKS"
POLYTRANS = "POLY"
UFS = "UFS"

# Combiner types
UTILITY_COMBINER = 'utility'
EVIDENCE_COMBINER = 'evidence'
TRAINABLE_COMBINER = 'trainable'

FILEPATH_MODELS = "files/models/"

SILHOUETTE = "SIL"
