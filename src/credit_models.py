"""
Модели PD, LGD-12 и LGD-24 на условном портфеле потребкредитов
с логированием в MLflow на DagsHub.

Модели:
  PD      — вероятность дефолта (90+ в первые 9 месяцев) по данным на дату выдачи.
  LGD-12  — потери за 12 мес. после дефолта: LGD12 = 1 − чистое возмещение за мес. 1–12 / EAD.
            Признаки — на дату дефолта (аналог модели «0 месяцев»).
  LGD-24  — модель на 12-м месяце после дефолта: прогноз чистого возмещения за мес. 13–24.
            LGD24 = 1 − факт возмещения за мес. 1–12 − прогноз за мес. 13–24 (аналог модели «12 месяцев»).
Чистое возмещение = collected − legal_fee − postage_fee.

Запуск:
  pip install -r requirements.txt
  # учётные данные — только через переменные окружения, не в коде:
  export DAGSHUB_USER=<логин>  DAGSHUB_REPO=<репозиторий>  DAGSHUB_TOKEN=<токен>
  python src/credit_models.py                 # данные по умолчанию: data/consumer_loans_portfolio.xlsx
  python src/credit_models.py --no-mlflow     # только обучение и метрики, без логирования
  python src/credit_models.py --local-mlflow  # логирование в локальную папку mlruns
"""
import argparse, os, json, tempfile, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

warnings.filterwarnings('ignore')
SEED = 42
REPORT = pd.Period('2026-10', 'M')
EXPERIMENT = 'conditional-portfolio-credit-risk'

APP_NUM = ['amount_or_limit', 'term_months', 'interest_rate', 'age', 'dependants', 'work_experience_months',
           'monthly_income', 'salary_client', 'credit_history_months', 'active_loans_cnt', 'other_loans_payment', 'pti']
APP_CAT = ['product', 'channel', 'sex', 'region', 'education', 'marital_status', 'employment', 'client_type', 'max_dpd_hist_12m']


# ------------------------------------------------------------------ data
def load(path):
    if path.endswith('.xlsx'):
        print('Читаю Excel (1–2 минуты)...')
        sh = pd.read_excel(path, sheet_name=['Займы', 'Помесячно', 'Взыскание'])
        L, M, C = sh['Займы'], sh['Помесячно'], sh['Взыскание']
    else:                                     # папка с loans.csv / monthly.csv / collection.csv
        L = pd.read_csv(os.path.join(path, 'loans.csv'))
        M = pd.read_csv(os.path.join(path, 'monthly.csv'))
        C = pd.read_csv(os.path.join(path, 'collection.csv'))
    L['term_months'] = L['term_months'].fillna(0)        # карта — без срока
    L['issue_month'] = L['issue_month'].astype(str)
    return L, M, C


def months_between(a, b):
    return (pd.Period(b, 'M') - pd.Period(a, 'M')).n


def build_pd(L):
    df = L.copy()
    df['target'] = df['default_9m'].astype(int)
    df['split'] = np.where(df['issue_month'] <= '2024-06', 'train', 'test')     # out-of-time
    return df


def pre_default_features(L, M):
    """Поведение займа до дефолта по помесячной таблице."""
    d = L[L['default_any'] == 1][['loan_id', 'default_mob']]
    m = M.merge(d, on='loan_id')
    m = m[m['mob'] <= m['default_mob']]
    g = m.groupby('loan_id').agg(
        months_before_default=('mob', 'max'),
        paid_before_default=('paid_total', 'sum'),
        sched_before_default=('scheduled_payment', 'sum'),
        months_with_payment=('paid_total', lambda x: int((x > 0).sum())),
        months_dpd_pos=('max_dpd_in_month', lambda x: int((x > 0).sum())))
    g['paid_to_sched'] = g['paid_before_default'] / g['sched_before_default'].replace(0, np.nan)
    return g.reset_index()


def build_lgd(L, M, C):
    d = L[L['default_any'] == 1].copy()
    d['obs_months'] = d['default_month'].astype(str).map(lambda x: months_between(x, str(REPORT)))
    d = d.merge(pre_default_features(L, M), on='loan_id', how='left')
    d['log_ead'] = np.log1p(d['ead'])
    d['ead_to_amount'] = d['ead'] / d['amount_or_limit']
    d['ead_to_income'] = d['ead'] / d['monthly_income']
    c = C.copy()
    c['net'] = c['collected'] - c['legal_fee'] - c['postage_fee']
    c = c.merge(d[['loan_id', 'ead']], on='loan_id')

    def window(a, b, col='net'):
        x = c[(c['months_since_default'] >= a) & (c['months_since_default'] <= b)]
        return x.groupby('loan_id')[col].sum()

    d['rr_0_12'] = d['loan_id'].map(window(1, 12)).fillna(0) / d['ead']
    d['rr_13_24'] = d['loan_id'].map(window(13, 24)).fillna(0) / d['ead']

    # признаки первого года взыскания (для LGD-24)
    y1 = c[c['months_since_default'] <= 12]
    f = y1.groupby('loan_id').agg(
        calls_12=('calls_cnt', 'sum'), sms_12=('sms_cnt', 'sum'), rpc_12=('right_party_contacts', 'sum'),
        promises_12=('promises_cnt', 'sum'), promises_sum_12=('promises_sum', 'sum'),
        pay_months_12=('collected', lambda x: int((x > 0).sum())),
        legal_cost_12=('legal_fee', 'sum'))
    last3 = y1[y1['months_since_default'] >= 10].groupby('loan_id')['collected'].sum().rename('collected_m10_12')
    garn = y1[y1['legal_stage'].astype(str).str.contains('удержания')].groupby('loan_id').size().rename('garnish_months_12')
    debt12 = y1[y1['months_since_default'] == 12].set_index('loan_id')['debt_end'].rename('debt_12')
    d = d.merge(f, on='loan_id', how='left').merge(last3, on='loan_id', how='left') \
         .merge(garn, on='loan_id', how='left').merge(debt12, on='loan_id', how='left')
    for col in ['collected_m10_12', 'garnish_months_12']:
        d[col] = d[col].fillna(0)
    d['debt12_to_ead'] = d['debt_12'] / d['ead']
    d['collected_m10_12_share'] = d['collected_m10_12'] / d['ead']
    lgd12 = d[d['obs_months'] >= 12].copy()
    lgd12['target'] = (1 - lgd12['rr_0_12']).clip(0, 1)
    # для LGD-24 нужны займы, не погашенные к 12 месяцу (иначе прогнозировать нечего)
    lgd24 = d[(d['obs_months'] >= 24) & (d['debt_12'].fillna(0) > 1)].copy()
    lgd24['target'] = lgd24['rr_13_24'].clip(0, 1)          # модель прогнозирует возмещение мес. 13–24
    for df in (lgd12, lgd24):                                # out-of-time: ранние 70 % дефолтов — обучение
        cut = df['default_month'].astype(str).sort_values().iloc[int(len(df) * 0.7)]
        df['split'] = np.where(df['default_month'].astype(str) < cut, 'train', 'test')
        df.attrs['oot_cut'] = cut
    return lgd12, lgd24


# ------------------------------------------------------------------ models
def preprocessor(num, cat, scale=True):
    num_steps = [('imp', SimpleImputer(strategy='median'))] + ([('sc', StandardScaler())] if scale else [])
    return ColumnTransformer([
        ('num', Pipeline(num_steps), num),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                          ('oh', OneHotEncoder(handle_unknown='ignore', min_frequency=20))]), cat)])


def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


def ks(y, p):
    df = pd.DataFrame({'y': y, 'p': p}).sort_values('p')
    cg = (1 - df['y']).cumsum() / (1 - df['y']).sum()
    cb = df['y'].cumsum() / df['y'].sum()
    return float((cg - cb).abs().max())


def psi(a, b, bins=10):
    q = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    ea = np.histogram(a, q)[0] / len(a) + 1e-6
    eb = np.histogram(np.clip(b, q[0], q[-1]), q)[0] / len(b) + 1e-6
    return float(np.sum((ea - eb) * np.log(ea / eb)))


def train_pd(df):
    num, cat = APP_NUM, APP_CAT
    tr, te = df[df.split == 'train'], df[df.split == 'test']
    cands = {
        'logreg': Pipeline([('prep', preprocessor(num, cat)),
                            ('clf', LogisticRegression(C=0.3, max_iter=2000, class_weight=None))]),
        'hgb': Pipeline([('prep', preprocessor(num, cat, scale=False)),
                         ('clf', HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                                                l2_regularization=1.0, min_samples_leaf=50, random_state=SEED))]),
    }
    res = {}
    for name, pipe in cands.items():
        pipe.fit(tr[num + cat], tr['target'])
        p_tr = pipe.predict_proba(tr[num + cat])[:, 1]; p_te = pipe.predict_proba(te[num + cat])[:, 1]
        res[name] = dict(model=pipe, p_tr=p_tr, p_te=p_te, metrics={
            'gini_train': gini(tr['target'], p_tr), 'gini_test': gini(te['target'], p_te),
            'ks_test': ks(te['target'].values, p_te), 'brier_test': brier_score_loss(te['target'], p_te),
            'pd_mean_pred_test': float(p_te.mean()), 'dr_actual_test': float(te['target'].mean()),
            'psi_score_train_test': psi(p_tr, p_te)})
    best = max(res, key=lambda k: res[k]['metrics']['gini_test'])
    return best, res, num + cat, tr, te


def lgd_features(kind):
    base_num = ['ead', 'log_ead', 'ead_to_amount', 'ead_to_income', 'default_mob', 'months_with_payment',
                'months_dpd_pos', 'paid_to_sched', 'interest_rate', 'age', 'monthly_income', 'work_experience_months',
                'salary_client', 'active_loans_cnt', 'pti']
    base_cat = ['product', 'channel', 'employment', 'region', 'education']
    if kind == 'lgd24':
        base_num = base_num + ['rr_0_12', 'debt12_to_ead', 'collected_m10_12_share', 'pay_months_12', 'garnish_months_12',
                               'calls_12', 'sms_12', 'rpc_12', 'promises_12', 'promises_sum_12', 'legal_cost_12']
    return base_num, base_cat


def train_lgd(df, kind):
    num, cat = lgd_features(kind)
    tr, te = df[df.split == 'train'], df[df.split == 'test']
    cands = {
        'ridge': Pipeline([('prep', preprocessor(num, cat)), ('reg', Ridge(alpha=5.0))]),
        'hgb': Pipeline([('prep', preprocessor(num, cat, scale=False)),
                         ('reg', HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                                               l2_regularization=1.0, min_samples_leaf=20, random_state=SEED))]),
    }
    res = {}
    for name, pipe in cands.items():
        pipe.fit(tr[num + cat], tr['target'])
        p_tr = np.clip(pipe.predict(tr[num + cat]), 0, 1); p_te = np.clip(pipe.predict(te[num + cat]), 0, 1)
        m = {'mae_train': mean_absolute_error(tr['target'], p_tr), 'mae_test': mean_absolute_error(te['target'], p_te),
             'rmse_test': float(np.sqrt(mean_squared_error(te['target'], p_te))), 'r2_test': r2_score(te['target'], p_te),
             'spearman_test': float(spearmanr(te['target'], p_te).correlation),
             # взвешенное по EAD смещение на уровне портфеля (прогноз − факт)
             'portfolio_bias_test': float(np.average(p_te, weights=te['ead']) - np.average(te['target'], weights=te['ead'])),
             'target_mean_test': float(np.average(te['target'], weights=te['ead']))}
        if kind == 'lgd24':
            lgd24_fact = (1 - te['rr_0_12'] - te['target']).clip(0, 1)
            lgd24_pred = (1 - te['rr_0_12'] - p_te).clip(0, 1)
            m['lgd24_mae_test'] = mean_absolute_error(lgd24_fact, lgd24_pred)
            m['lgd24_fact_ead_weighted'] = float(np.average(lgd24_fact, weights=te['ead']))
            m['lgd24_pred_ead_weighted'] = float(np.average(lgd24_pred, weights=te['ead']))
        res[name] = dict(model=pipe, p_tr=p_tr, p_te=p_te, metrics=m)
    best = min(res, key=lambda k: res[k]['metrics']['mae_test'])
    return best, res, num + cat, tr, te


# ------------------------------------------------------------------ plots
def plots_pd(te, p, out):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(te['target'], p)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], '--', color='grey'); ax[0].set_title(f'ROC (тест), Gini = {gini(te["target"], p):.2f}')
    ax[0].set_xlabel('FPR'); ax[0].set_ylabel('TPR')
    q = pd.qcut(p, 10, labels=False, duplicates='drop')
    cal = pd.DataFrame({'q': q, 'p': p, 'y': te['target'].values}).groupby('q').mean()
    ax[1].plot(cal['p'], cal['y'], 'o-'); lim = max(cal['p'].max(), cal['y'].max()) * 1.1
    ax[1].plot([0, lim], [0, lim], '--', color='grey'); ax[1].set_title('Калибровка по децилям (тест)')
    ax[1].set_xlabel('Прогноз PD'); ax[1].set_ylabel('Факт DR')
    fig.tight_layout(); path = os.path.join(out, 'pd_roc_calibration.png'); fig.savefig(path, dpi=120); plt.close(fig)
    return [path]


def plots_lgd(te, p, kind, out):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(te['target'], p, s=10, alpha=0.5); ax.plot([0, 1], [0, 1], '--', color='grey')
    ax.set_xlabel('Факт'); ax.set_ylabel('Прогноз')
    ax.set_title({'lgd12': 'LGD-12: факт vs прогноз (тест)', 'lgd24': 'Возмещение мес. 13–24: факт vs прогноз (тест)'}[kind])
    fig.tight_layout(); path = os.path.join(out, f'{kind}_fact_vs_pred.png'); fig.savefig(path, dpi=120); plt.close(fig)
    return [path]


def importance(model, X, y, out, name, scoring):
    r = permutation_importance(model, X, y, n_repeats=5, random_state=SEED, scoring=scoring)
    imp = pd.DataFrame({'feature': X.columns, 'importance': r.importances_mean}).sort_values('importance', ascending=False)
    path = os.path.join(out, f'{name}_feature_importance.csv'); imp.to_csv(path, index=False)
    fig, ax = plt.subplots(figsize=(6, 5)); top = imp.head(15)[::-1]
    ax.barh(top['feature'], top['importance']); ax.set_title(f'{name}: важность признаков (перестановки, тест)')
    fig.tight_layout(); p2 = os.path.join(out, f'{name}_feature_importance.png'); fig.savefig(p2, dpi=120); plt.close(fig)
    return [path, p2]


# ------------------------------------------------------------------ mlflow
def setup_mlflow(local):
    import mlflow
    if local:
        mlflow.set_tracking_uri((ROOT / 'mlruns').as_uri())
    else:
        user, repo, token = os.environ.get('DAGSHUB_USER'), os.environ.get('DAGSHUB_REPO'), os.environ.get('DAGSHUB_TOKEN')
        if not (user and repo and token):
            raise SystemExit('Задайте переменные окружения DAGSHUB_USER, DAGSHUB_REPO, DAGSHUB_TOKEN (или запустите с --local-mlflow).')
        os.environ['MLFLOW_TRACKING_USERNAME'] = user
        os.environ['MLFLOW_TRACKING_PASSWORD'] = token
        mlflow.set_tracking_uri(f'https://dagshub.com/{user}/{repo}.mlflow')
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


def log_run(mlflow, name, best, res, features, tr, te, artifacts, params, register):
    from mlflow.models import infer_signature
    with mlflow.start_run(run_name=f'{name}-{best}'):
        mlflow.set_tags({'model_family': name, 'algorithm': best, 'data': 'synthetic conditional portfolio',
                         'oot_split': params.get('oot_split', '')})
        mlflow.log_params({**params, 'algorithm': best, 'n_features': len(features), 'n_train': len(tr), 'n_test': len(te)})
        for cand, r in res.items():                           # метрики всех кандидатов — для сравнения
            mlflow.log_metrics({f'{cand}__{k}': float(v) for k, v in r['metrics'].items()})
        mlflow.log_metrics({k: float(v) for k, v in res[best]['metrics'].items()})
        for a in artifacts:
            mlflow.log_artifact(a)
        X = te[features].head(5)
        model = res[best]['model']
        pred = model.predict_proba(X)[:, 1] if name == 'pd' else model.predict(X)
        sig = infer_signature(X, pred)
        mlflow.sklearn.log_model(sk_model=model, name=f'{name}_model', signature=sig, input_example=X,
                                 registered_model_name=(f'{name}_model' if register else None))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(ROOT / 'data' / 'consumer_loans_portfolio.xlsx'), help='Excel-файл портфеля или папка с CSV')
    ap.add_argument('--no-mlflow', action='store_true')
    ap.add_argument('--local-mlflow', action='store_true')
    ap.add_argument('--no-register', action='store_true', help='не регистрировать модели в Model Registry')
    a = ap.parse_args()

    L, M, C = load(a.data)
    out = str(ROOT / 'outputs'); os.makedirs(out, exist_ok=True)
    summary = {}
    runs = []

    # PD
    pdf = build_pd(L)
    best, res, feats, tr, te = train_pd(pdf)
    arts = plots_pd(te, res[best]['p_te'], out) + importance(res[best]['model'], te[feats], te['target'], out, 'pd', 'roc_auc')
    summary['pd'] = {'best': best, **{k: round(v, 4) for k, v in res[best]['metrics'].items()}}
    runs.append(('pd', best, res, feats, tr, te, arts, {'target': 'default_9m (90+ в первые 9 мес.)', 'oot_split': 'выдачи до 2024-06 / после'}))

    # LGD
    lgd12, lgd24 = build_lgd(L, M, C)
    for kind, df, tgt in [('lgd12', lgd12, 'LGD12 = 1 − чистое возмещение мес. 1–12 / EAD'),
                          ('lgd24', lgd24, 'чистое возмещение мес. 13–24 / EAD (LGD24 = 1 − RR12 − прогноз)')]:
        best, res, feats, tr, te = train_lgd(df, kind)
        arts = plots_lgd(te, res[best]['p_te'], kind, out) + importance(res[best]['model'], te[feats], te['target'], out, kind, 'neg_mean_absolute_error')
        summary[kind] = {'best': best, **{k: round(v, 4) for k, v in res[best]['metrics'].items()}}
        runs.append((kind, best, res, feats, tr, te, arts, {'target': tgt, 'oot_split': f"дефолты до {df.attrs['oot_cut']} / с {df.attrs['oot_cut']}"}))

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    with open(os.path.join(out, 'summary.json'), 'w', encoding='utf8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('Артефакты:', out)

    if a.no_mlflow:
        return
    mlflow = setup_mlflow(a.local_mlflow)
    for r in runs:
        log_run(mlflow, *r, register=not a.no_register)
    print('Готово: модели залогированы в эксперимент', EXPERIMENT)


if __name__ == '__main__':
    main()
