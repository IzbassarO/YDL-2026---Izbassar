import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from scipy.stats import loguniform, randint
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
Xraw=fe(tr[feat])
# pre-impute once (unsupervised -> no label leak) for fast tuning
imp=IterativeImputer(estimator=BayesianRidge(),max_iter=5,random_state=42)
X=pd.DataFrame(imp.fit_transform(Xraw),columns=Xraw.columns)
cv42=StratifiedKFold(5,shuffle=True,random_state=42)
def sc(m):return Pipeline([('s',StandardScaler()),('m',m)])
P("Шаг 2: перетюнинг на импутированном пространстве (cv seed42, f1_macro)")
# SVC grid
gsv=GridSearchCV(sc(SVC(random_state=42)),{'m__C':[50,80,120,200],'m__gamma':[0.005,0.008,0.012,0.02]},scoring='f1_macro',cv=cv42,n_jobs=-1).fit(X,y)
P(f"  SVC best={gsv.best_score_:.4f} {gsv.best_params_} (старое C=80,g=0.008)")
# HGB random search
hgb=RandomizedSearchCV(HistGradientBoostingClassifier(random_state=42),
    {'learning_rate':loguniform(0.02,0.2),'max_iter':randint(200,500),'max_leaf_nodes':randint(20,63),
     'min_samples_leaf':randint(15,60),'l2_regularization':loguniform(0.1,10)},
    n_iter=25,scoring='f1_macro',cv=cv42,n_jobs=-1,random_state=42).fit(X,y)
P(f"  HGB best={hgb.best_score_:.4f}")
P("   "+str({k:(round(v,4) if isinstance(v,float) else v) for k,v in hgb.best_params_.items()}))
# MLP random search
mlp=RandomizedSearchCV(sc(MLPClassifier(max_iter=500,early_stopping=True,random_state=42)),
    {'m__hidden_layer_sizes':[(128,64),(256,128),(128,128,64),(200,100),(256,128,64),(150,)],
     'm__alpha':loguniform(1e-5,1e-1),'m__learning_rate_init':loguniform(1e-4,5e-3),'m__activation':['relu','tanh']},
    n_iter=20,scoring='f1_macro',cv=cv42,n_jobs=-1,random_state=42).fit(X,y)
P(f"  MLP best={mlp.best_score_:.4f}")
P("   "+str({k:(round(v,5) if isinstance(v,float) else v) for k,v in mlp.best_params_.items()}))
# ExtraTrees small search
et=RandomizedSearchCV(ExtraTreesClassifier(random_state=42,n_jobs=-1),
    {'n_estimators':randint(400,800),'max_features':loguniform(0.4,0.95),'min_samples_leaf':randint(1,6)},
    n_iter=12,scoring='f1_macro',cv=cv42,n_jobs=-1,random_state=42).fit(X,y)
P(f"  ExtraTrees best={et.best_score_:.4f} {dict((k,round(v,3) if isinstance(v,float) else v) for k,v in et.best_params_.items())}")
open(".step2_out.txt","w").write("\n".join(log)); print("DONE")
