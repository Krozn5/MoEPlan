import os, pandas as pd, psycopg2, json
from tqdm import tqdm

database = 'tpcds1gb'
user = 'postgres'
host = '162.105.86.21'
password = '123456'
port = '5435'
simple_queries = list(range(1, 100))
# simple_queries = [3, 7, 12, 20, 26, 37, 42, 43, 50, 55, 62, 84, 91, 96, 99, 18, 27, 52, 82, 98]

def get_query():
	queries = []
	with psycopg2.connect(host=host, port=port, database=database, user=user, password=password) as conn:
		with conn.cursor() as cur:
			cur.execute('SELECT * FROM public.queries')
			q_list = list(cur.fetchall())
			q_list = sorted(q_list, key=lambda x:x[0])
			for q in tqdm(q_list):
				# print(q[0])
				if q[0] in simple_queries:
					query = q[3].split('\n')
					query = filter(lambda x:not x.startswith('--'), query)
					query = ' '.join(query)
					queries.append(query)
	return queries

def get_train_plan():
	ids, jsons = [], []
	with psycopg2.connect(host=host, port=port, database=database, user=user, password=password) as conn:
		with conn.cursor() as cur:
			cur.execute('SELECT * FROM public.queries')
			q_list = list(cur.fetchall())
			q_list = sorted(q_list, key=lambda x:x[0])
			counter = 0
			for q in tqdm(q_list):
				# print(q[0])
				if q[0] in simple_queries:
					ids.append(counter)
					counter += 1
					cur.execute('explain (format json, analyze, buffers, verbose) ' + q[3][:q[3].find(';')+1])
					jsons.append(json.dumps(cur.fetchall()[0][0][0]))
	return pd.DataFrame({'id': ids, 'json': jsons})

def extract_plan(plan):
    table, join, predicate = [], [], []
    keys = plan.keys()
    if 'Relation Name' in plan:
        table.append(plan['Relation Name']+' '+plan['Alias'])
    for key in keys:
        if 'Cond' in key and 'Join' in plan['Node Type']:
            join.append(plan[key][1:-1].replace(' ', ''))
        if 'Filter' in key and 'by' not in key:
            predicate.append(plan[key])
    if 'Plans' in keys:
        for child in plan['Plans']:
            t, j, p = extract_plan(child)
            table += t
            join += j
            predicate += p
    return table, join, predicate

def find_parens(s):
    print('find_parens', s)
    toret = {}
    pstack = []
    for i, c in enumerate(s):
        if c == '(':
            pstack.append(i)
        elif c == ')':
            if len(pstack) == 0:
                raise IndexError("No matching closing parens at: " + str(i))
            toret[pstack.pop()] = i
    if len(pstack) > 0:
        raise IndexError("No matching opening parens at: " + str(pstack.pop()))
    return toret

def find_parens_with_exp(s, limit=4):
    for j in range(limit):
        ss = s[j:]
        try:
            parens = find_parens(ss)
            break
        except Exception as e:
            pass
    else:
        for j in range(limit):
            ss = s[:-j]
            try:
                parens = find_parens(ss)
                break
            except Exception as e:
                pass
    return ss, parens

def extract_predicates(predicates):
    ps = []
    for i, plist in tqdm(enumerate(predicates)):
        predicate, ppredicate = [], []
        for p in plist:
            p_parens = find_parens(p)
            while 0 in p_parens and p_parens[0] == len(p) - 1:
                p = p[1:-1]
                p_parens = find_parens(p)
            try:
                for pp in p.split('AND'):
                    # pp_parens = find_parens(pp)
                    pp, pp_parens = find_parens_with_exp(pp.strip())
                    while 0 in pp_parens and pp_parens[0] == len(pp) - 1:
                        pp = pp[1:-1]
                        pp_parens = find_parens(pp)
                    ppredicate += [s.strip() for s in pp.split('OR')]
            except Exception as e:
                for pp in p.split('OR'):
                    # pp_parens = find_parens(pp)
                    pp, pp_parens = find_parens_with_exp(pp.strip())
                    while 0 in pp_parens and pp_parens[0] == len(pp) - 1:
                        pp = pp[1:-1]
                        pp_parens = find_parens(pp)
                    ppredicate += [s.strip() for s in pp.split('AND')]
        for p in ppredicate:
            p_parens = find_parens(p)
            while 0 in p_parens and p_parens[0] == len(p) - 1:
                p = p[1:-1]
                p_parens = find_parens(p)
            for op in ['>=', '<=', '~~', '>', '<', '=']:
                if p.find(' '+op+' ') != -1:
                    break
            else:
                print(f'Not supported query {i+1}')
                continue
                # raise Exception(f'Unrecognized op in {p}')
            p = p.split(' '+op+' ')
            predicate += [p[0], op, p[1]]
        ps.append(',,'.join(predicate))
    return ps

# synthetic.sql
query_path = 'workloads'
if not os.path.exists(query_path):
    os.makedirs(query_path)
queries = get_query()
with open(f'{query_path}/synthetic.sql', 'w') as f:
    f.write('\n'.join(queries))

# train_plan.csv
plan_path = 'plan_and_cost'
if not os.path.exists(plan_path):
    os.makedirs(plan_path)
df = get_train_plan()
df.to_csv(f'{plan_path}/train_plan.csv', index=False, sep=',')

# synthetic.csv
query_path = 'workloads'
plan_path = 'plan_and_cost'
df = pd.read_csv(f'{plan_path}/train_plan.csv', sep=',')
tables, joins, predicates, cards = [], [], [], []
for index, row in tqdm(list(df.iterrows())):
	plan = json.loads(row['json'])
	table, join, predicate = extract_plan(plan=plan['Plan'])
	tables.append(','.join(table))
	joins.append(','.join(join))
	predicates.append(predicate)
	cards.append(plan['Plan']['Actual Rows'])
predicates = extract_predicates(predicates)
df = pd.DataFrame({'table':tables, 'join':joins, 'predicate':predicates, 'card':cards})
df.to_csv(f'{query_path}/synthetic.csv', index=False, sep='#', header=None)
