import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
log=[]; P=lambda s:(log.append(str(s)),print(s,flush=True))
tr=pd.read_csv('train.csv'); feat=[c for c in tr.columns if c not in('id','sleep_stage')]; y=tr.sleep_stage.values
EEG=['eeg_delta_power','eeg_theta_power','eeg_alpha_power','eeg_sigma_power','eeg_beta_power','eeg_gamma_power']
def fe(df):
    X=df.copy(); tot=X[EEG].clip(lower=0).sum(1)+1e-6
    for b in EEG: X['rel_'+b]=X[b]/tot
    X['delta_beta']=X['eeg_delta_power']/(X['eeg_beta_power'].abs()+1e-6)
    X['theta_alpha']=X['eeg_theta_power']/(X['eeg_alpha_power'].abs()+1e-6)
    X['slow_dom']=X['eeg_slow_osc_power']+X['eeg_delta_power']
    X['eog_burst_missing']=df['eog_burst_index'].isna().astype(int)
    return X
X=fe(tr[feat])
def sc(m):return Pipeline([('s',StandardScaler()),('m',m)])
def II():return IterativeImputer(estimator=BayesianRidge(),max_iter=5,random_state=42)
# members
def mem(seed,which):
    d={
    'svc_old':('svc',sc(SVC(C=80,gamma=0.008,probability=True,random_state=seed))),
    'svc_new':('svc',sc(SVC(C=50,gamma=0.005,probability=True,random_state=seed))),
    'hgb_old':('hgb',HistGradientBoostingClassifier(random_state=seed,learning_rate=0.079,max_iter=240,max_leaf_nodes=43,min_samples_leaf=24,l2_regularization=7.26)),
    'hgb_new':('hgb',HistGradientBoostingClassifier(random_state=seed,learning_rate=0.021,max_iter=457,max_leaf_nodes=43,min_samples_leaf=58,l2_regularization=2.607)),
    'et_old':('et',ExtraTreesClassifier(n_estimators=430,max_features=0.89,min_samples_leaf=1,random_state=seed,n_jobs=-1)),
    'et_new':('et',ExtraTreesClassifier(n_estimators=530,max_features=0.624,min_samples_leaf=1,random_state=seed,n_jobs=-1)),
    'mlp1':('mlp1',sc(MLPClassifier(hidden_layer_sizes=(128,64),alpha=1e-3,max_iter=400,early_stopping=True,random_state=seed))),
    'mlp2_old':('mlp2',sc(MLPClassifier(hidden_layer_sizes=(200,100),activation='tanh',alpha=1e-3,max_iter=400,early_stopping=True,random_state=seed))),
    'mlp2_new':('mlp2',sc(MLPClassifier(hidden_layer_sizes=(256,128,64),activation='tanh',alpha=0.0028,learning_rate_init=0.00054,max_iter=500,early_stopping=True,random_state=seed))),
    }
    return d[which]
def vote(keys,seed):
    return Pipeline([('imp',II()),('v',VotingClassifier([mem(seed,k) for k in keys],voting='soft',n_jobs=-1))])
def stack(keys,seed):
    return Pipeline([('imp',II()),('s',StackingClassifier([mem(seed,k) for k in keys],final_estimator=LogisticRegression(max_iter=2000),cv=5,n_jobs=-1))])
def msc(builder,keys):
    return np.mean([cross_val_score(builder(keys,s),X,y,cv=StratifiedKFold(5,shuffle=True,random_state=s),scoring='f1_macro',n_jobs=-1).mean() for s in [0,1,42]]),\
           np.std([cross_val_score(builder(keys,s),X,y,cv=StratifiedKFold(5,shuffle=True,random_state=s),scoring='f1_macro',n_jobs=-1).mean() for s in [0,1,42]])
configs=[
 ('V5 current (baseline)',vote,['svc_old','hgb_old','et_old','mlp1','mlp2_old']),
 ('V5 retuned',vote,['svc_new','hgb_new','et_new','mlp1','mlp2_new']),
 ('V6 old+new mlp2',vote,['svc_new','hgb_new','et_new','mlp1','mlp2_old','mlp2_new']),
 ('Stack5 retuned',stack,['svc_new','hgb_new','et_new','mlp1','mlp2_new']),
]
best=None
for nm,b,keys in configs:
    mu,sd=msc(b,keys); P(f"{nm:24s} {mu:.4f} ± {sd:.4f}")
    if best is None or mu>best[1]: best=(nm,mu)
P(f"\nЛучший: {best[0]} = {best[1]:.4f}  (текущий рекорд 0.8440)")
open(".step3_out.txt","w").write("\n".join(log)); print("DONE")
