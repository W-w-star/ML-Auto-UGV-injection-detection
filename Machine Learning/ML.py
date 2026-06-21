import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

TEST_SIZE = 0.3              
RANDOM_STATE = 42            

RF_N_ESTIMATORS = 287
RF_MAX_DEPTH = 25
RF_MIN_SAMPLES_SPLIT = 10
RF_MIN_SAMPLES_LEAF = 1
RF_MAX_FEATURES = 'sqrt'
RF_BOOTSTRAP = False
RF_CCP_ALPHA = 0.0
RF_CRITERION = 'gini'

KNN_N_NEIGHBORS = 3
KNN_WEIGHTS = 'distance'
KNN_METRIC = 'euclidean'
KNN_ALGORITHM = 'auto'
KNN_LEAF_SIZE = 30
KNN_P = 2

MLP_HIDDEN_LAYERS = (128, 64)
MLP_ACTIVATION = 'relu'
MLP_SOLVER = 'adam'
MLP_ALPHA = 0.000291063591313307
MLP_LEARNING_RATE_INIT = 0.0030049873591901574
MLP_MAX_ITER = 173
MLP_MOMENTUM = 0.9535954961421276 
MLP_LEARNING_RATE = 'constant'
MLP_EARLY_STOPPING = True

DT_MAX_DEPTH = 88
DT_MIN_SAMPLES_SPLIT = 2
DT_MIN_SAMPLES_LEAF = 2
DT_MAX_FEATURES = 1.0
DT_CRITERION = 'gini'
DT_SPLITTER = 'best'
DT_CCP_ALPHA = 0.0

SVM_C = 82.78543388872865
SVM_GAMMA = 2.075176455915079
SVM_KERNEL = 'rbf'
SVM_SHRINKING = True
SVM_PROBABILITY = False
SVM_TOL = 1e-3
SVM_MAX_ITER = -1

print("="*80)
print(" Machine Learning Full Parameter and Academic Dual-Level Indicator Output Framework ")
print("="*80)

df = pd.read_csv('/.csv')

if 'Tt' in df.columns:
    df = df.drop(columns=['Tt'])
    
features = [col for col in df.columns if col not in ['Ts', 'label', 'trajectory_label']]

X = df[features].astype(np.float32)
y = df['label'].astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "RF": RandomForestClassifier(
        bootstrap=RF_BOOTSTRAP, ccp_alpha=RF_CCP_ALPHA, criterion=RF_CRITERION,
        max_depth=RF_MAX_DEPTH, min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        min_samples_split=RF_MIN_SAMPLES_SPLIT, n_estimators=RF_N_ESTIMATORS,
        max_features=RF_MAX_FEATURES, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "KNN": KNeighborsClassifier(
        algorithm=KNN_ALGORITHM, leaf_size=KNN_LEAF_SIZE, metric=KNN_METRIC,
        n_neighbors=KNN_N_NEIGHBORS, p=KNN_P, weights=KNN_WEIGHTS, n_jobs=-1
    ),
    "MLP": MLPClassifier(
        activation=MLP_ACTIVATION, alpha=MLP_ALPHA, hidden_layer_sizes=MLP_HIDDEN_LAYERS,
        max_iter=MLP_MAX_ITER, solver=MLP_SOLVER, momentum=MLP_MOMENTUM,
        learning_rate_init=MLP_LEARNING_RATE_INIT, learning_rate=MLP_LEARNING_RATE,
        early_stopping=MLP_EARLY_STOPPING, random_state=RANDOM_STATE
    ),
    "DT": DecisionTreeClassifier(
        ccp_alpha=DT_CCP_ALPHA, criterion=DT_CRITERION, max_depth=DT_MAX_DEPTH,
        min_samples_leaf=DT_MIN_SAMPLES_LEAF, min_samples_split=DT_MIN_SAMPLES_SPLIT,
        max_features=DT_MAX_FEATURES, splitter=DT_SPLITTER, random_state=RANDOM_STATE
    ),
    "SVM": SVC(
        C=SVM_C, kernel=SVM_KERNEL,  gamma=SVM_GAMMA,
        shrinking=SVM_SHRINKING, probability=SVM_PROBABILITY,
        tol=SVM_TOL, max_iter=SVM_MAX_ITER, random_state=RANDOM_STATE
    )
}

LABEL_MAPPING = {0: 'Auth', 1: 'PM', 2: 'VD'}
classes = np.unique(y_test)
class_names = [LABEL_MAPPING.get(int(c), str(c)) for c in classes]

print("\n[1] Training models, please wait...")

table_rows = []
cms = {}

for name, model in models.items():
    t0 = time.perf_counter()
    model.fit(X_train_scaled, y_train)
    tt_ms = (time.perf_counter() - t0) * 1000  

    t1 = time.perf_counter()
    preds = model.predict(X_test_scaled)
    pt_ms = (time.perf_counter() - t1) * 1000 

    va = accuracy_score(y_test, preds) * 100
    
    pr = precision_score(y_test, preds, average='macro', zero_division=0) * 100
    rc = recall_score(y_test, preds, average='macro', zero_division=0) * 100
    
    if (pr + rc) > 0:
        fs = (2 * pr * rc) / (pr + rc)
    else:
        fs = 0.0

    cm = confusion_matrix(y_test, preds, labels=classes)
    cms[name] = cm
    
    row_data = [name, va, pr, rc, fs]

    dr_list, mdr_list, far_list = [], [], []
    
    for i, cls in enumerate(classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)

        DR = (TP / (TP + FN) * 100) if (TP + FN) > 0 else 0.0
        MDR = 100.0 - DR
        FAR = (FP / (FP + TN) * 100) if (FP + TN) > 0 else 0.0
        
        dr_list.append(DR)
        mdr_list.append(MDR)
        far_list.append(FAR)

    row_data.extend(dr_list)
    row_data.extend(mdr_list)
    row_data.extend(far_list)
    row_data.extend([tt_ms, pt_ms])
    
    table_rows.append(row_data)

print("\n[2] Generating Confusion Matrices visualization...")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Confusion Matrices for the Developed ML Classifiers', fontsize=18, fontweight='bold', y=0.95)
axes = axes.ravel() 

for idx, (name, cm) in enumerate(cms.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14})
    axes[idx].set_title(f'{name}', fontsize=14, pad=10)
    axes[idx].set_xlabel('Predicted Label', fontsize=12)
    axes[idx].set_ylabel('True Label', fontsize=12)

axes[5].axis('off')
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()

print("\n[3] Academic Data Tables:")
header_tuples = [('Classifier', '')]
header_tuples += [('Avg. Performance Metrics', metric) for metric in ['VA', 'PR', 'RC', 'FS']]
for metric_type in ['Class-specific DR (%)', 'Class-specific MDR (%)', 'Class-specific FAR (%)']:
    for cls in class_names:
        header_tuples.append((metric_type, cls))
header_tuples += [('Time (ms)', 'TT'), ('Time (ms)', 'PT')]

multi_columns = pd.MultiIndex.from_tuples(header_tuples)
df_results = pd.DataFrame(table_rows, columns=multi_columns)

cols_part1 = [('Classifier', '')] + \
             [('Avg. Performance Metrics', m) for m in ['VA', 'PR', 'RC', 'FS']] + \
             [('Time (ms)', m) for m in ['TT', 'PT']]
df_part1 = df_results[cols_part1]

print("\n>>> Table 1: Overall Average Metrics and Time Overhead")
print("-" * 80)
print(df_part1.to_string(index=False, float_format="%.2f"))
print("-" * 80)

cols_part2 = [('Classifier', '')] + [col for col in df_results.columns if 'Class-specific' in col[0]]
df_part2 = df_results[cols_part2]

print("\n>>> Table 2: Class-specific Detailed Metrics (DR, MDR, FAR)")
print("-" * 100)
print(df_part2.to_string(index=False, float_format="%.2f"))
print("-" * 100)