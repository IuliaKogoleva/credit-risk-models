"""
Генератор условного портфеля потребительских кредитов.
Продукты: POS-кредит, кредитная карта, кредит наличными (онлайн / офлайн).
Таблицы: loans (заёмщик + займ), monthly (помесячные платежи и макс. просрочка),
collection (софт и лигал взыскание после дефолта).
Дефолт: 90+ дней просрочки в первые 9 месяцев с даты выдачи (default_9m).
Все займы, достигшие 90+, передаются на взыскание: 1-й месяц после дефолта — софт,
со 2-го месяца — лигал.
"""
import numpy as np, pandas as pd, sys, runpy
from pathlib import Path
DATA = Path(__file__).resolve().parents[1] / 'data'
DATA.mkdir(exist_ok=True)

SEED = 2026
rng = np.random.default_rng(SEED)
N_PER_MONTH = 500
ISSUE_MONTHS = pd.period_range('2023-01', '2024-12', freq='M')
REPORT = pd.Period('2026-10', freq='M')          # последний наблюдаемый месяц
BV = 45.0                                         # условная базовая величина, BYN

REGIONS = ['г. Минск', 'Минская', 'Гомельская', 'Брестская', 'Витебская', 'Гродненская', 'Могилёвская']
REG_P = [0.30, 0.12, 0.14, 0.14, 0.11, 0.10, 0.09]
REG_INC = dict(zip(REGIONS, [1.30, 1.05, 0.92, 0.92, 0.88, 0.92, 0.90]))
EMP = ['Гос. предприятие', 'Частная компания', 'ИП / самозанятый', 'Пенсионер', 'Не работает']
EMP_P = [0.44, 0.40, 0.08, 0.05, 0.03]
EDU = ['Высшее', 'Среднее специальное', 'Среднее']
EDU_P = [0.45, 0.40, 0.15]
MAR = ['Женат/замужем', 'Холост/не замужем', 'Разведён(а)', 'Вдовец/вдова']
MAR_P = [0.52, 0.30, 0.15, 0.03]

PRODUCTS = {
    # product: share, channels{name:p}, base pd9 (logit shift), recovery multiplier
    'POS-кредит':          dict(share=0.30, ch={'Офлайн (магазин)': 0.80, 'Онлайн (маркетплейс)': 0.20}, pd9=0.040, rec=0.80),
    'Кредитная карта':     dict(share=0.20, ch={'Офлайн (отделение)': 0.55, 'Онлайн': 0.45},              pd9=0.035, rec=0.80),
    'Наличные офлайн':     dict(share=0.30, ch={'Офлайн (отделение)': 1.0},                               pd9=0.022, rec=1.15),
    'Наличные онлайн':     dict(share=0.20, ch={'Онлайн': 1.0},                                           pd9=0.060, rec=0.85),
}

def choice(a, p, size=None):
    return rng.choice(a, p=np.array(p) / np.sum(p), size=size)

# ---------------------------------------------------------------- borrowers & loans
rows = []
lid = 100000
for ip in ISSUE_MONTHS:
    for _ in range(N_PER_MONTH):
        lid += 1
        prod = choice(list(PRODUCTS), [v['share'] for v in PRODUCTS.values()])
        P = PRODUCTS[prod]
        ch = choice(list(P['ch']), list(P['ch'].values()))
        online = 'Онлайн' in ch
        age = int(np.clip(rng.normal(37 if online else 41, 11), 21, 70))
        sex = choice(['М', 'Ж'], [0.47, 0.53])
        reg = choice(REGIONS, REG_P)
        emp = choice(EMP, EMP_P) if age < 60 else choice(EMP, [0.20, 0.15, 0.05, 0.55, 0.05])
        edu = choice(EDU, EDU_P)
        mar = choice(MAR, MAR_P)
        dep = int(choice([0, 1, 2, 3], [0.40, 0.30, 0.24, 0.06]))
        inc = float(np.round(rng.lognormal(np.log(1750), 0.45) * REG_INC[reg] * (0.75 if emp == 'Пенсионер' else 0.5 if emp == 'Не работает' else 1.0), -1))
        exp_m = 0 if emp in ('Пенсионер', 'Не работает') else int(np.clip(rng.exponential(55), 3, 360))
        salary_client = int(rng.random() < (0.35 if emp == 'Гос. предприятие' else 0.15))
        repeat = int(rng.random() < 0.35)
        hist_m = int(np.clip(rng.exponential(60) + (24 if repeat else 0), 0, 300))
        act_loans = int(choice([0, 1, 2, 3, 4], [0.38, 0.30, 0.18, 0.09, 0.05]))
        hist_dpd = choice(['0', '1-30', '31-60', '61+'], [0.78, 0.14, 0.05, 0.03]) if hist_m > 0 else 'Нет истории'

        # loan terms
        if prod == 'POS-кредит':
            amt = float(np.round(np.clip(rng.lognormal(np.log(1100), 0.6), 200, 6000), -1))
            term = int(choice([6, 10, 12, 18, 24], [0.20, 0.15, 0.35, 0.15, 0.15]))
            rate = 0.0 if rng.random() < 0.45 else float(np.round(rng.uniform(0.16, 0.22), 3))
        elif prod == 'Кредитная карта':
            amt = float(np.round(np.clip(rng.lognormal(np.log(1800), 0.55), 300, 8000), -2))   # лимит
            term = 0
            rate = float(np.round(rng.uniform(0.22, 0.30), 3))
        elif prod == 'Наличные офлайн':
            amt = float(np.round(np.clip(rng.lognormal(np.log(6000), 0.7), 1000, 40000), -2))
            term = int(choice([12, 24, 36, 48, 60], [0.10, 0.25, 0.30, 0.15, 0.20]))
            rate = float(np.round(rng.uniform(0.14, 0.18), 3))
        else:
            amt = float(np.round(np.clip(rng.lognormal(np.log(2500), 0.6), 500, 15000), -2))
            term = int(choice([6, 12, 18, 24, 36], [0.10, 0.30, 0.20, 0.25, 0.15]))
            rate = float(np.round(rng.uniform(0.18, 0.26), 3))

        r = rate / 12
        if term:
            pmt = amt / term if r == 0 else amt * r / (1 - (1 + r) ** -term)
        else:
            pmt = np.nan
        ext_pmt = float(np.round(act_loans * rng.uniform(80, 300), -1))
        pti = ((pmt if term else 0.05 * amt) + ext_pmt) / max(inc, 1)

        # risk score: logit of 9m default
        lg = np.log(P['pd9'] / (1 - P['pd9']))
        lg += 0.9 * (pti - 0.30) + {'0': -0.25, '1-30': 0.45, '31-60': 0.95, '61+': 1.4, 'Нет истории': 0.35}[hist_dpd]
        lg += -0.015 * (age - 38) + (0.25 if sex == 'М' else -0.15) + {'Высшее': -0.30, 'Среднее специальное': 0.10, 'Среднее': 0.35}[edu]
        lg += {'Гос. предприятие': -0.10, 'Частная компания': 0.05, 'ИП / самозанятый': 0.30, 'Пенсионер': -0.20, 'Не работает': 0.90}[emp]
        lg += -0.45 * salary_client - 0.35 * repeat + 0.12 * act_loans - 0.003 * min(exp_m, 120) + 0.08 * dep
        lg += rng.normal(0, 0.45)                                    # необъяснимая часть
        pd9 = 1 / (1 + np.exp(-lg))
        score = int(np.clip(np.round(620 - 55 * (lg - np.log(0.04 / 0.96)) + rng.normal(0, 25)), 300, 900))
        grade = 'A' if score >= 700 else 'B' if score >= 650 else 'C' if score >= 600 else 'D' if score >= 550 else 'E'

        issue_date = ip.start_time + pd.Timedelta(days=int(rng.integers(0, 28)))
        rows.append(dict(loan_id=lid, issue_date=issue_date.date(), issue_month=str(ip), product=prod, channel=ch,
                         amount_or_limit=amt, term_months=term if term else None, interest_rate=rate,
                         monthly_payment=round(pmt, 2) if term else None,
                         age=age, sex=sex, region=reg, education=edu, marital_status=mar, dependants=dep,
                         employment=emp, work_experience_months=exp_m, monthly_income=inc, salary_client=salary_client,
                         client_type='Повторный' if repeat else 'Новый', credit_history_months=hist_m,
                         active_loans_cnt=act_loans, other_loans_payment=ext_pmt, pti=round(pti, 3),
                         max_dpd_hist_12m=hist_dpd, score=score, risk_grade=grade, _pd9=pd9, _rec=P['rec']))
loans = pd.DataFrame(rows)

# ---------------------------------------------------------------- monthly performance
mrows, crows = [], []
dinfo = {}

def legal_and_soft(L, D_mob, D_period, ead, collected_before=0.0):
    """Взыскание после дефолта. D_mob — месяц жизни, в котором впервые 90+."""
    debt = ead
    emp, inc = L['employment'], L['monthly_income']
    working = emp not in ('Не работает',)
    # индивидуальная «взыскиваемость»
    base = {'Гос. предприятие': 1.0, 'Частная компания': 0.85, 'ИП / самозанятый': 0.6, 'Пенсионер': 0.9, 'Не работает': 0.15}[emp]
    base *= L['_rec'] * np.exp(rng.normal(0, 0.5))
    payer_type = rng.random()
    out_m, out_c = [], []
    k = 0
    dpd = 90 + int(rng.integers(1, 29))
    closed = False
    while True:
        k += 1
        per = D_period + k
        if per > REPORT or closed or k > 48:
            break
        dpd += 30 + int(rng.integers(-1, 2))
        if k == 1:                                               # софт
            stage = 'Софт'
            calls = int(rng.integers(12, 45)); auto = int(rng.integers(5, 25)); sms = int(rng.integers(4, 16))
            rpc = int(rng.binomial(calls, 0.12 if working else 0.07)); prom = int(rng.binomial(max(rpc, 0), 0.35))
            prom_sum = round(prom * debt * rng.uniform(0.03, 0.12), 2)
            if payer_type < 0.035 * base:
                pay = debt                                      # полное погашение на софте
            elif rng.random() < 0.38 * min(base, 1.6):
                pay = debt * rng.uniform(0.02, 0.10)
            else:
                pay = 0.0
            legal_stage, lfee, post = None, 0.0, round(sms * 0.05, 2)
        else:                                                    # лигал
            stage = 'Лигал'
            calls = int(rng.integers(0, 6)); auto = int(rng.integers(0, 4)); sms = int(rng.integers(1, 5))
            rpc = int(rng.binomial(calls, 0.10)); prom = int(rng.binomial(rpc, 0.30)); prom_sum = round(prom * debt * rng.uniform(0.02, 0.06), 2)
            if k == 2:
                legal_stage = 'Претензия, исполнительная надпись нотариуса'
                lfee = round(BV * rng.uniform(0.8, 1.3), 2); post = round(rng.uniform(3, 6), 2)
                pay = debt * rng.uniform(0.01, 0.06) if rng.random() < 0.25 * base else 0.0
            elif k <= 4:
                legal_stage = 'Передано в органы принудительного исполнения'
                lfee = round(rng.uniform(0, 8), 2) if k == 3 else 0.0; post = round(rng.uniform(1, 4), 2)
                pay = debt * rng.uniform(0.01, 0.05) if rng.random() < 0.30 * base else 0.0
            else:
                garnish = working and rng.random() < min(0.70, 0.45 * base)
                legal_stage = 'Принудительное исполнение: удержания' if garnish else 'Принудительное исполнение: поиск имущества/дохода'
                lfee = 0.0; post = round(rng.uniform(0, 2), 2)
                if garnish:
                    pay = min(debt, inc * rng.uniform(0.05, 0.15) * min(1.0, 0.5 + 0.3 * base))
                else:
                    pay = debt * rng.uniform(0.005, 0.03) if rng.random() < 0.10 * base else 0.0
            if k == 2 and rng.random() < 0.02 * base:
                pay = debt
        pay = float(round(min(pay, debt), 2))
        penalty = round(debt * 0.0 , 2)
        debt_before = debt
        debt = round(debt - pay, 2)
        if debt <= 1.0:
            debt = 0.0; closed = True
        out_c.append(dict(loan_id=L['loan_id'], report_month=str(per), months_since_default=k, collection_stage=stage,
                          legal_stage=legal_stage, calls_cnt=calls, autodialer_cnt=auto, sms_cnt=sms,
                          right_party_contacts=rpc, promises_cnt=prom, promises_sum=prom_sum,
                          debt_start=round(debt_before, 2), collected=pay, legal_fee=lfee, postage_fee=post,
                          debt_end=debt))
        out_m.append(dict(loan_id=L['loan_id'], report_month=str(per), mob=D_mob + k, status=stage + (' (погашен)' if closed else ''),
                          scheduled_payment=0.0, paid_total=pay, principal_paid=None, interest_paid=None,
                          principal_balance=None, total_debt=debt, max_dpd_in_month=0 if closed else min(dpd, 9999)))
    return out_m, out_c

for L in loans.to_dict('records'):
    ip = pd.Period(L['issue_month'], freq='M')
    max_mob = (REPORT - ip).n
    pd9 = L['_pd9']
    u = rng.random()
    if u < pd9:
        fate = 'default9'; m0 = int(choice([1, 2, 3, 4, 5, 6], [0.24, 0.22, 0.18, 0.15, 0.12, 0.09]))
    elif u < pd9 * 1.55:
        fate = 'default_late'; m0 = int(rng.integers(7, 30))
    elif u < pd9 * 1.55 + 0.10 + 2.0 * pd9:
        fate = 'transient'; m0 = int(rng.integers(1, 20)); miss_len = int(choice([1, 2], [0.7, 0.3]))
    else:
        fate = 'good'; m0 = 10 ** 6
    prepay_m = int(rng.integers(3, 40)) if (fate in ('good', 'transient') and rng.random() < (0.30 if L['product'] != 'Кредитная карта' else 0.0)) else 10 ** 6
    r = L['interest_rate'] / 12
    is_card = L['product'] == 'Кредитная карта'
    bal = 0.0 if is_card else L['amount_or_limit']
    term = L['term_months'] or 0
    pmt = L['monthly_payment'] or 0.0
    arrears = 0.0; missed = 0; dpd_prev = 0; fees = 0.0
    card_util = rng.uniform(0.3, 0.9) if fate not in ('default9','default_late') else rng.uniform(0.7, 1.0)
    revolver = rng.random() < 0.6 or fate in ('default9','default_late')
    closed = False
    for mob in range(1, max_mob + 1):
        per = ip + mob
        # ---- schedule
        if is_card:
            spend = L['amount_or_limit'] * card_util * rng.uniform(0.05, 0.35) if mob <= 1 or rng.random() < 0.8 else 0.0
            if mob == 1:
                spend = L['amount_or_limit'] * card_util * rng.uniform(0.4, 0.8)
            spend = max(0.0, min(spend, L['amount_or_limit'] - bal))
            interest = bal * r if revolver or missed else 0.0
            bal_due = bal + interest
            sched = 0.0 if bal_due <= 0 else max(min(bal_due, 15.0), 0.05 * bal_due) if revolver else bal_due
        else:
            if bal <= 0.005:
                closed = True
            interest = bal * r
            sched = 0.0 if closed else min(pmt, bal + interest)
            spend = 0.0
        if closed and not is_card:
            break
        # ---- behaviour
        in_miss = (fate in ('default9', 'default_late') and mob >= m0) or (fate == 'transient' and m0 <= mob < m0 + miss_len)
        if in_miss:
            partial = sched * rng.uniform(0.1, 0.5) if (rng.random() < 0.15 and missed < 2) else 0.0
            paid = partial
            arrears += sched - partial
            missed += 1
            fees += round(sched * 0.01, 2)
        elif fate == 'transient' and mob == m0 + miss_len and missed > 0:
            paid = sched + arrears + fees; arrears = 0.0; missed_before = missed; missed = 0; fees = 0.0
        else:
            paid = sched
        if mob == prepay_m and not is_card and not in_miss:
            paid = bal + interest + arrears
            arrears = 0.0; missed = 0
        if is_card:
            if not in_miss and not revolver:
                paid = bal_due
            elif not in_miss:
                paid = max(sched, bal_due * rng.uniform(0.05, 0.4)) if rng.random() < 0.5 else sched
            int_paid = min(paid, interest); prin_paid = paid - int_paid
            bal = max(0.0, bal + interest - paid) + spend
            total_debt = bal + fees
        else:
            int_paid = min(paid, interest)
            prin_paid = max(0.0, paid - int_paid - (fees if paid > sched else 0))
            bal = max(0.0, bal + interest - paid + (fees if paid > sched else 0)) if not in_miss else bal + interest - paid
            total_debt = bal + fees
        # ---- max dpd in month
        if missed > 0:
            dpd = (missed - 1) * 30 + int(rng.integers(5, 29)) + (int(rng.integers(0, 3)))
        elif fate == 'transient' and mob == m0 + miss_len:
            dpd = (missed_before - 1) * 30 + int(rng.integers(5, 29))        # макс. просрочка до погашения
        else:
            dpd = 0 if rng.random() > 0.04 else int(rng.integers(1, 6))      # технические просрочки
        status = 'Текущий' if dpd == 0 else 'Просрочка'
        if not is_card and bal <= 0.005 and missed == 0:
            status = 'Погашен досрочно' if mob < term else 'Погашен'
        mrows.append(dict(loan_id=L['loan_id'], report_month=str(per), mob=mob, status=status,
                          scheduled_payment=round(sched, 2), paid_total=round(paid, 2), principal_paid=round(prin_paid, 2),
                          interest_paid=round(int_paid, 2), principal_balance=round(bal, 2), total_debt=round(total_debt, 2),
                          max_dpd_in_month=dpd))
        if status.startswith('Погашен'):
            break
        if dpd >= 90:                                                     # дефолт → взыскание
            L_ead = round(total_debt, 2)
            dinfo[L['loan_id']] = (mob, str(per), L_ead)
            om, oc = legal_and_soft(L, mob, per, L_ead)
            mrows.extend(om); crows.extend(oc)
            break

monthly = pd.DataFrame(mrows)
coll = pd.DataFrame(crows)
di = pd.DataFrame([(k,)+v for k,v in dinfo.items()], columns=['loan_id','default_mob','default_month','ead'])
loans = loans.merge(di, on='loan_id', how='left')
loans['default_9m'] = ((loans['default_mob'].notna()) & (loans['default_mob'] <= 9)).astype(int)
loans['default_any'] = loans['default_mob'].notna().astype(int)
loans = loans.drop(columns=['_pd9', '_rec'])
loans.to_pickle(DATA / 'loans.pkl'); monthly.to_pickle(DATA / 'monthly.pkl'); coll.to_pickle(DATA / 'coll.pkl')
print('займов:', len(loans), '| строк помесячно:', len(monthly), '| строк взыскания:', len(coll))

# сборка Excel
runpy.run_path(str(Path(__file__).with_name('build_xlsx.py')), run_name='__main__')
