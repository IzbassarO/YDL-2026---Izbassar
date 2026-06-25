import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier,
                              ExtraTreesRegressor, HistGradientBoostingRegressor)
log=[]; P=lambda s:(log.append(str(s)),print(s,flush=True))
tr=pd.read_csv('train.csv'); te=pd.read_csv('test.csv'); feat=[c for c in tr.columns if c not in('id','sleep_stage')]; y=tr.sleep_stage.values
names={0:'Wake',1:'Light',2:'Deep',3:'REM'}; has=tr.eog_burst_index.notna().values
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

# --- (a) OOF R2 of eog recovery for each regressor (on observed rows) ---
P("Шаг 1a: OOF R2 восстановления eog_burst_index (на наблюдаемых строках):")
Xobs=X[has].drop(columns=['eog_burst_index']).fillna(X.median()); yobs=tr.eog_burst_index[has].values
regs={'BayesianRidge':BayesianRidge(),
      'KNN':KNeighborsRegressor(n_neighbors=15),
      'ExtraTrees':ExtraTreesRegressor(n_estimators=300,n_jobs=-1,random_state=42),
      'HGB':HistGradientBoostingRegressor(random_state=42)}
for nm,r in regs.items():
    r2=cross_val_score(r,Xobs,yobs,cv=5,scoring='r2',n_jobs=-1).mean()
    P(f"   {nm:14s} R2={r2:.3f}")

# --- (b) V5 CV f1_macro multi-seed with each IterativeImputer estimator (shared imputer, fit once/fold) ---
def voting(seed):
    sc=lambda m:Pipeline([('s',StandardScaler()),('m',m)])
    return VotingClassifier([
        ('svc',sc(SVC(C=80,gamma=0.008,probability=True,random_state=seed))),
        ('hgb',HistGradientBoostingClassifier(random_state=seed,learning_rate=0.079,max_iter=240,max_leaf_nodes=43,min_samples_leaf=24,l2_regularization=7.26)),
        ('et',ExtraTreesClassifier(n_estimators=430,max_features=0.89,min_samples_leaf=1,random_state=seed,n_jobs=-1)),
        ('mlp1',sc(MLPClassifier(hidden_layer_sizes=(128,64),alpha=1e-3,max_iter=400,early_stopping=True,random_state=seed))),
        ('mlp2',sc(MLPClassifier(hidden_layer_sizes=(200,100),activation='tanh',alpha=1e-3,max_iter=400,early_stopping=True,random_state=seed)))],
        voting='soft',n_jobs=-1)
def full(est_factory,seed):
    return Pipeline([('imp',IterativeImputer(estimator=est_factory(),max_iter=5,random_state=42)),('vote',voting(seed))])
imps={'BayesianRidge(текущий)':lambda:BayesianRidge(),
      'ExtraTreesReg':lambda:ExtraTreesRegressor(n_estimators=200,n_jobs=-1,random_state=42),
      'HGBReg':lambda:HistGradientBoostingRegressor(random_state=42)}
P("\nШаг 1b: V5 CV f1_macro (multi-seed 0,1,42) по импьютеру:")
results={}
for nm,fac in imps.items():
    scs=[cross_val_score(full(fac,s),X,y,cv=StratifiedKFold(5,shuffle=True,random_state=s),scoring='f1_macro',n_jobs=-1).mean() for s in [0,1,42]]
    results[nm]=np.mean(scs); P(f"   {nm:24s} {np.mean(scs):.4f} ± {np.std(scs):.4f}  (median baseline=0.8365, BayesRidge-II=0.8440)")
    # per-class + no-eog for best-ish
    oof=cross_val_predict(full(fac,42),X,y,cv=StratifiedKFold(5,shuffle=True,random_state=42),n_jobs=-1)
    P(f"       per-class: "+" ".join(f"{names[i]}={np.round(__import__('sklearn.metrics',fromlist=['f1_score']).f1_score(y,oof,average=None)[i],3)}" for i in range(4))+
      f" | no-eog={__import__('sklearn.metrics',fromlist=['f1_score']).f1_score(y[~has],oof[~has],average='macro'):.4f}")
best=max(results,key=results.get); P(f"\nЛучший импьютер: {best} = {results[best]:.4f}")
open(".step1_out.txt","w").write("\n".join(log)); print("DONE")
