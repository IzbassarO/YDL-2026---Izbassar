import warnings; warnings.filterwarnings("ignore")
import os, sys, json
os.chdir("/Users/izbassar/Documents/Projects/YDL/2 Week/4-5 Day Kaggle Competition")
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
import lightgbm as lgb
from catboost import CatBoostClassifier

_tr=pd.read_csv("train.csv"); _feat=[c for c in _tr.columns if c not in("id","sleep_stage")]
Y=_tr.sleep_stage.values
EEG=["eeg_delta_power","eeg_theta_power","eeg_alpha_power","eeg_sigma_power","eeg_beta_power","eeg_gamma_power"]
def fe(df):
    X=df.copy(); tot=X[EEG].clip(lower=0).sum(1)+1e-6
    for b in EEG: X["rel_"+b]=X[b]/tot
    X["delta_beta"]=X["eeg_delta_power"]/(X["eeg_beta_power"].abs()+1e-6)
    X["theta_alpha"]=X["eeg_theta_power"]/(X["eeg_alpha_power"].abs()+1e-6)
    X["slow_dom"]=X["eeg_slow_osc_power"]+X["eeg_delta_power"]
    X["eog_burst_missing"]=df["eog_burst_index"].isna().astype(int)
    return X
X=fe(_tr[_feat])
def II(): return IterativeImputer(estimator=BayesianRidge(),max_iter=5,random_state=42)
def sc(m): return Pipeline([("s",StandardScaler()),("m",m)])

def make(cfg, seed):
    t=cfg["type"]; p=cfg.get("params",{})
    if t=="lgbm":
        d=dict(objective="multiclass",num_class=4,n_estimators=400,learning_rate=0.03,num_leaves=31,
               subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5.0,reg_alpha=0.0,
               min_child_samples=30,max_depth=-1,n_jobs=1,verbosity=-1,random_state=seed)
        d.update(p); return lgb.LGBMClassifier(**d)
    if t=="cat":
        d=dict(iterations=600,learning_rate=0.03,depth=6,l2_leaf_reg=5.0,loss_function="MultiClass",
               random_seed=seed,thread_count=1,verbose=0,allow_writing_files=False)
        d.update(p); return CatBoostClassifier(**d)
    if t=="hgb":
        d=dict(random_state=seed,learning_rate=0.079,max_iter=240,max_leaf_nodes=43,min_samples_leaf=24,l2_regularization=7.26)
        d.update(p); return HistGradientBoostingClassifier(**d)
    if t=="svc": return sc(SVC(C=80,gamma=0.008,probability=True,random_state=seed))
    if t=="et": return ExtraTreesClassifier(n_estimators=430,max_features=0.89,min_samples_leaf=1,random_state=seed,n_jobs=1)
    if t=="mlp1": return sc(MLPClassifier(hidden_layer_sizes=(128,64),alpha=1e-3,max_iter=400,early_stopping=True,random_state=seed))
    if t=="mlp2": return sc(MLPClassifier(hidden_layer_sizes=(200,100),activation="tanh",alpha=1e-3,max_iter=400,early_stopping=True,random_state=seed))
    raise ValueError(t)

def pipe(cfg, seed):
    t=cfg["type"]
    if t=="ensemble":
        members=[(m if isinstance(m,str) else m["type"]+str(i), make({"type":m} if isinstance(m,str) else m, seed)) for i,m in enumerate(cfg["members"])]
        head=VotingClassifier(members,voting="soft",weights=cfg.get("weights"),n_jobs=1)
        return Pipeline([("imp",II()),("v",head)])
    if t=="stack":
        members=[(m if isinstance(m,str) else m["type"]+str(i), make({"type":m} if isinstance(m,str) else m, seed)) for i,m in enumerate(cfg["members"])]
        return Pipeline([("imp",II()),("s",StackingClassifier(members,final_estimator=LogisticRegression(max_iter=2000),cv=5,n_jobs=1))])
    return Pipeline([("imp",II()),("m",make(cfg,seed))])

CVJOBS=int(os.environ.get("CV_JOBS","-1"))
def evaluate(cfg, seeds=(0,1,42)):
    s=[]
    for sd in seeds:
        sc_=cross_val_score(pipe(cfg,sd),X,Y,cv=StratifiedKFold(5,shuffle=True,random_state=sd),scoring="f1_macro",n_jobs=CVJOBS).mean()
        s.append(sc_)
    return float(np.mean(s)), float(np.std(s))

def build_submission(cfg, out, seeds=(0,1,2,3,42)):
    _te=pd.read_csv("test.csv"); Xt=fe(_te[_feat]); names={0:"Wake",1:"Light",2:"Deep",3:"REM"}
    probs=np.zeros((len(Xt),4))
    for sd in seeds:
        m=pipe(cfg,sd); m.fit(X,Y); probs+=m.predict_proba(Xt)
    pred=(probs/len(seeds)).argmax(1)
    sub=pd.DataFrame({"id":_te.id,"sleep_stage":pred}); sub.to_csv(out,index=False)
    ss=pd.read_csv("sample_submission.csv")
    ok=list(sub.columns)==list(ss.columns) and len(sub)==len(ss) and bool((sub.id.values==ss.id.values).all()) and set(sub.sleep_stage)<=set([0,1,2,3]) and bool(sub.sleep_stage.notna().all())
    dist={names[i]:round(float((pred==i).mean()*100),1) for i in range(4)}
    print(json.dumps({"out":out,"format_ok":ok,"dist":dist}))

if __name__=="__main__":
    if sys.argv[1]=="submit":
        cfg=json.loads(sys.argv[2]); out=sys.argv[3]
        seeds=tuple(json.loads(sys.argv[4])) if len(sys.argv)>4 else (0,1,2,3,42)
        build_submission(cfg,out,seeds)
    else:
        cfg=json.loads(sys.argv[1])
        seeds=tuple(json.loads(sys.argv[2])) if len(sys.argv)>2 else (0,1,42)
        mu,sd=evaluate(cfg,seeds)
        print(json.dumps({"label":cfg.get("label",cfg.get("type")),"cv":round(mu,4),"std":round(sd,4)}))
