import pandas as pd, numpy as np
from pathlib import Path
DATA = Path(__file__).resolve().parents[1] / 'data'
OUT = DATA / 'consumer_loans_portfolio.xlsx'
L = pd.read_pickle(DATA / 'loans.pkl'); M = pd.read_pickle(DATA / 'monthly.pkl'); C = pd.read_pickle(DATA / 'coll.pkl')
REPORT = pd.Period('2026-10', 'M')

L = L.copy()
L['issue_date'] = pd.to_datetime(L['issue_date'])
loans_cols = ['loan_id', 'issue_date', 'issue_month', 'product', 'channel', 'amount_or_limit', 'term_months', 'interest_rate',
              'monthly_payment', 'age', 'sex', 'region', 'education', 'marital_status', 'dependants', 'employment',
              'work_experience_months', 'monthly_income', 'salary_client', 'client_type', 'credit_history_months',
              'active_loans_cnt', 'other_loans_payment', 'pti', 'max_dpd_hist_12m', 'score', 'risk_grade',
              'default_9m', 'default_any', 'default_mob', 'default_month', 'ead']
L = L[loans_cols]

# ---- summary tables
s1 = L.groupby(['product']).agg(loans=('loan_id', 'size'), volume=('amount_or_limit', 'sum'), avg_rate=('interest_rate', 'mean'),
                                 default_9m=('default_9m', 'mean'), default_any=('default_any', 'mean')).reset_index()
s2 = L.groupby(['product', 'channel']).agg(loans=('loan_id', 'size'), default_9m=('default_9m', 'mean')).reset_index()
s3 = L.groupby('risk_grade').agg(loans=('loan_id', 'size'), avg_score=('score', 'mean'), default_9m=('default_9m', 'mean')).reset_index()
d = L[L.default_any == 1].copy()
d['obs'] = d.default_month.map(lambda m: (REPORT - pd.Period(m, 'M')).n)
C2 = C.merge(d[['loan_id', 'product', 'ead', 'obs']], on='loan_id')
curve = []
for prod in ['Все продукты'] + sorted(d['product'].unique()):
    dd = d if prod == 'Все продукты' else d[d['product'] == prod]
    row = {'product': prod}
    for K in (1, 3, 6, 9, 12, 18, 24, 30):
        ids = dd[dd.obs >= K].loan_id
        x = C2[C2.loan_id.isin(ids) & (C2.months_since_default <= K)]
        e = dd[dd.loan_id.isin(ids)].ead.sum()
        row[f'RR_{K}m'] = x.collected.sum() / e if e > 0 and len(ids) >= 20 else np.nan
    curve.append(row)
curve = pd.DataFrame(curve)

DICT = [
    ('Займы', 'loan_id', 'Идентификатор займа'),
    ('Займы', 'issue_date / issue_month', 'Дата и месяц выдачи (выдачи с 01.2023 по 12.2024, по 500 займов в месяц)'),
    ('Займы', 'product', 'Продукт: POS-кредит, кредитная карта, наличные офлайн, наличные онлайн'),
    ('Займы', 'channel', 'Канал продаж: офлайн (магазин, отделение) или онлайн (маркетплейс, сайт/приложение)'),
    ('Займы', 'amount_or_limit', 'Сумма займа, BYN; для карты — кредитный лимит'),
    ('Займы', 'term_months', 'Срок, мес.; для карты пусто (возобновляемый лимит)'),
    ('Займы', 'interest_rate', 'Годовая ставка: POS 0 % (рассрочка) или 16–22 %; карты 22–30 %; наличные офлайн 14–18 %; онлайн 18–26 %'),
    ('Займы', 'monthly_payment', 'Аннуитетный платёж по графику, BYN (для карты — пусто, минимальный платёж 5 % от задолженности)'),
    ('Займы', 'age … max_dpd_hist_12m', 'Характеристики заёмщика на дату выдачи: возраст, пол, регион, образование, семейное положение, иждивенцы, занятость, стаж, доход, зарплатный клиент, новый/повторный, длина кредитной истории, число действующих кредитов и платежи по ним, PTI (платёжная нагрузка), макс. просрочка за 12 мес. по истории'),
    ('Займы', 'score / risk_grade', 'Условный скоринговый балл (300–900) и грейд A–E'),
    ('Займы', 'default_9m', '1 — займ достиг просрочки 90+ в течение первых 9 месяцев с даты выдачи (определение дефолта)'),
    ('Займы', 'default_any', '1 — займ достиг 90+ в любой момент наблюдения (включая позже 9 мес.)'),
    ('Займы', 'default_mob / default_month', 'Месяц жизни займа и календарный месяц, в котором впервые 90+'),
    ('Займы', 'ead', 'Задолженность на момент дефолта (остаток основного долга + начисленные непогашенные проценты + штрафы), BYN'),
    ('Помесячно', 'report_month / mob', 'Отчётный месяц и месяц жизни займа (1 = первый месяц после выдачи). Наблюдение до 10.2026'),
    ('Помесячно', 'status', 'Текущий / Просрочка / Погашен / Погашен досрочно / Софт / Лигал (после дефолта)'),
    ('Помесячно', 'scheduled_payment', 'Платёж к оплате в месяце по графику (для карты — минимальный или полный)'),
    ('Помесячно', 'paid_total / principal_paid / interest_paid', 'Фактически оплачено в месяце, в т. ч. основной долг и проценты. После дефолта — только paid_total (взыскание)'),
    ('Помесячно', 'principal_balance / total_debt', 'Остаток основного долга и общая задолженность на конец месяца'),
    ('Помесячно', 'max_dpd_in_month', 'Максимальное количество дней просрочки в течение месяца'),
    ('Взыскание', 'months_since_default', 'Месяц после дефолта: 1 — софт-взыскание; со 2-го месяца долг передан на лигал'),
    ('Взыскание', 'collection_stage / legal_stage', 'Стадия (Софт/Лигал) и этап судебного/исполнительного взыскания'),
    ('Взыскание', 'calls_cnt / autodialer_cnt / sms_cnt', 'Звонки операторов, автоинформатор, SMS за месяц'),
    ('Взыскание', 'right_party_contacts', 'Успешные контакты с должником'),
    ('Взыскание', 'promises_cnt / promises_sum', 'Обещания оплатить: количество и сумма'),
    ('Взыскание', 'debt_start / collected / debt_end', 'Долг на начало месяца, взыскано за месяц, долг на конец месяца'),
    ('Взыскание', 'legal_fee / postage_fee', 'Расходы на взыскание: нотариальный тариф/сборы, почтовые расходы'),
]

with pd.ExcelWriter(OUT, engine='xlsxwriter', datetime_format='dd.mm.yyyy', date_format='dd.mm.yyyy') as xw:
    wb = xw.book
    hdr = wb.add_format({'bold': True, 'bg_color': '#E7E6E6', 'border': 1, 'text_wrap': True, 'valign': 'top'})
    wrap = wb.add_format({'text_wrap': True, 'valign': 'top'})
    bold = wb.add_format({'bold': True, 'font_size': 13})
    pct = wb.add_format({'num_format': '0.0%'})
    money = wb.add_format({'num_format': '#,##0.00'})

    # Описание
    ws = wb.add_worksheet('Описание'); xw.sheets['Описание'] = ws
    ws.set_column(0, 0, 14); ws.set_column(1, 1, 38); ws.set_column(2, 2, 95)
    ws.write(0, 0, 'Условный портфель потребительских кредитов (синтетические данные)', bold)
    notes = [
        'Данные полностью синтетические, сгенерированы для методических расчётов; не относятся к какому-либо банку. Суммы в BYN.',
        f'12 000 займов, выдачи 01.2023–12.2024; наблюдение до 10.2026 включительно. Поздние выдачи и поздние дефолты наблюдаются не до конца (цензурирование).',
        'Дефолт: просрочка 90+ дней в первые 9 месяцев с даты выдачи (default_9m). Займы, достигшие 90+ позже, также передаются на взыскание (default_any).',
        'После дефолта: 1-й месяц — софт-взыскание (звонки, SMS, обещания); со 2-го месяца долг передан на лигал (исполнительная надпись нотариуса, органы принудительного исполнения, удержания из дохода).',
        'После дефолта проценты и штрафы не начисляются; взысканные суммы уменьшают задолженность на дату дефолта (EAD). Чистое возмещение = collected − legal_fee − postage_fee.',
        'Листы: «Займы» — заёмщик и займ; «Помесячно» — платежи и максимальная просрочка по месяцам; «Взыскание» — софт и лигал по месяцам после дефолта; «Сводка» — уровни дефолтов и кривые возмещения.',
    ]
    for i, n in enumerate(notes):
        ws.write(2 + i, 0, '•'); ws.merge_range(2 + i, 1, 2 + i, 2, n, wrap); ws.set_row(2 + i, 32)
    r0 = 2 + len(notes) + 1
    for j, h in enumerate(['Лист', 'Поле', 'Описание']):
        ws.write(r0, j, h, hdr)
    for i, (a, b, c) in enumerate(DICT):
        ws.write(r0 + 1 + i, 0, a, wrap); ws.write(r0 + 1 + i, 1, b, wrap); ws.write(r0 + 1 + i, 2, c, wrap)
        if len(c) > 95: ws.set_row(r0 + 1 + i, 15 * (len(c) // 95 + 1))

    def put(df, name, widths=None, fmts=None):
        df.to_excel(xw, sheet_name=name, index=False)
        w = xw.sheets[name]
        for j, c in enumerate(df.columns):
            w.write(0, j, c, hdr)
            w.set_column(j, j, (widths or {}).get(c, max(12, min(28, len(c) + 2))), (fmts or {}).get(c))
        w.freeze_panes(1, 1); w.autofilter(0, 0, len(df), len(df.columns) - 1)

    put(L, 'Займы', {'issue_date': 12, 'product': 18, 'channel': 22, 'region': 14, 'employment': 20, 'education': 20, 'marital_status': 18},
        {'interest_rate': pct, 'pti': pct, 'monthly_payment': money, 'ead': money})
    put(M, 'Помесячно', {'status': 18}, {c: money for c in ['scheduled_payment', 'paid_total', 'principal_paid', 'interest_paid', 'principal_balance', 'total_debt']})
    put(C, 'Взыскание', {'legal_stage': 44, 'collection_stage': 12}, {c: money for c in ['promises_sum', 'debt_start', 'collected', 'legal_fee', 'postage_fee', 'debt_end']})

    ws = wb.add_worksheet('Сводка'); xw.sheets['Сводка'] = ws
    row = 0
    for title, df, pcols in [('Уровень дефолтов по продуктам', s1, ['avg_rate', 'default_9m', 'default_any']),
                             ('Уровень дефолтов по продуктам и каналам', s2, ['default_9m']),
                             ('Уровень дефолтов по грейдам', s3, ['default_9m']),
                             ('Накопленное возмещение (доля EAD) по месяцам после дефолта — по займам, наблюдаемым не менее K месяцев', curve, [c for c in curve.columns if c.startswith('RR')])]:
        ws.write(row, 0, title, bold); row += 1
        for j, c in enumerate(df.columns): ws.write(row, j, c, hdr)
        for i, rec in enumerate(df.itertuples(index=False)):
            for j, v in enumerate(rec):
                c = df.columns[j]
                if isinstance(v, float) and np.isnan(v): ws.write_blank(row + 1 + i, j, None)
                elif c in pcols: ws.write_number(row + 1 + i, j, float(v), pct)
                elif isinstance(v, (float, np.floating)): ws.write_number(row + 1 + i, j, float(v), money)
                elif isinstance(v, (int, np.integer)): ws.write_number(row + 1 + i, j, int(v))
                else: ws.write(row + 1 + i, j, str(v))
        row += len(df) + 3
    ws.set_column(0, 0, 22); ws.set_column(1, 9, 14)
print('Excel сохранён:', OUT)
