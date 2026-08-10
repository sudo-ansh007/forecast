import warnings, time, io, contextlib; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from make_dataset import encode

tr = pd.read_parquet("train.parquet").sort_values("created_date").reset_index(drop=True)
cut = int(len(tr)*0.8); a, b = tr.iloc[:cut], tr.iloc[cut:]
with contextlib.redirect_stdout(io.StringIO()): X, Xt = encode(a, b)
ya, yb = a.is_won.to_numpy(), b.is_won.to_numpy()
skf = lambda: StratifiedKFold(3, shuffle=True, random_state=0)
cal = lambda m: CalibratedClassifierCV(m, method="sigmoid", cv=skf())

def hgb(s, **kw):
    p = dict(max_depth=3, min_samples_leaf=50, max_iter=200, learning_rate=0.05,
             early_stopping=True, validation_fraction=0.15, random_state=s); p.update(kw)
    return HistGradientBoostingClassifier(**p)

MODELS = {
 "logreg (not a tree)":        lambda s: cal(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))),
 "single tree depth3":         lambda s: cal(DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=s)),
 "single tree unlimited":      lambda s: cal(DecisionTreeClassifier(random_state=s)),
 "random forest 500":          lambda s: cal(RandomForestClassifier(n_estimators=500, min_samples_leaf=50, random_state=s, n_jobs=-1)),
 "HGB raw (uncalibrated)":     lambda s: hgb(s),
 "HGB+sigmoid  SHIPPING":      lambda s: cal(hgb(s)),
 "HGB+isotonic":               lambda s: CalibratedClassifierCV(hgb(s), method="isotonic", cv=skf()),
 "HGB depth8 leaf5":           lambda s: cal(hgb(s, max_depth=8, min_samples_leaf=5)),
 "LightGBM (real)":            lambda s: cal(LGBMClassifier(max_depth=3, num_leaves=8, min_child_samples=50,
                                       n_estimators=200, learning_rate=0.05, random_state=s, verbose=-1, n_jobs=-1)),
 "XGBoost":                    lambda s: cal(XGBClassifier(max_depth=3, min_child_weight=50, n_estimators=200,
                                       learning_rate=0.05, random_state=s, eval_metric="logloss", n_jobs=-1)),
 "LightGBM scale_pos_weight":  lambda s: cal(LGBMClassifier(max_depth=3, num_leaves=8, min_child_samples=50,
                                       n_estimators=200, learning_rate=0.05, random_state=s, verbose=-1,
                                       n_jobs=-1, scale_pos_weight=(ya==0).sum()/(ya==1).sum())),
}
print(f"train {len(a):,}/{ya.sum()} wins   test {len(b):,}/{yb.sum()} wins   5 seeds each")
print(f"random-guess PR-AUC = {yb.mean():.4f}\n")
print(f"{'model':28}{'PR-AUC':>18}{'Brier':>9}{'ROC':>8}{'sum(p)/act':>12}{'fit s':>8}")
print("-"*83)
rows=[]
for name, f in MODELS.items():
    pr,br,rc,ra,ts = [],[],[],[],[]
    for s in range(5):
        t0=time.time(); m=f(s).fit(X,ya); ts.append(time.time()-t0)
        p=m.predict_proba(Xt)[:,1]
        pr.append(average_precision_score(yb,p)); br.append(brier_score_loss(yb,p))
        rc.append(roc_auc_score(yb,p)); ra.append(p.sum()/yb.sum())
    rows.append((name,np.mean(pr)))
    print(f"{name:28}{np.mean(pr):.4f} +-{np.std(pr):.4f}{np.mean(br):>9.5f}{np.mean(rc):>8.4f}{np.mean(ra):>11.2f}x{np.mean(ts):>8.2f}")
print("-"*83)
best=max(rows,key=lambda r:r[1]); ship=dict(rows)["HGB+sigmoid  SHIPPING"]
print(f"best PR-AUC: {best[0]} ({best[1]:.4f})   shipping: {ship:.4f}   gap {best[1]-ship:+.4f}")
