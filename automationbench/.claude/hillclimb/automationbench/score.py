import json,glob,sys,statistics as st
def load(v,s):
    return {json.load(open(f))['task_name']:json.load(open(f)) for f in glob.glob(f'{v}/{s}/*.json') if not f.endswith(('config.json','summary.json'))}
def sc(r): return .8*r['partial_credit']+.2*r['task_completed_correctly']
for v in sys.argv[1:]:
    for s in ['train','test']:
        R=load(v,s); x=list(R.values())
        print(v,s,len(x),'err',sum(bool(r['error']) for r in x),'score %.3f pc %.3f pass %.3f'%(st.mean(map(sc,x)),st.mean(r['partial_credit'] for r in x),st.mean(r['task_completed_correctly'] for r in x)),'in_tok %.2fM out %.0fk lat %.0f calls %.1f'%(sum(r['usage'].get('input_tokens',0) for r in x)/1e6,sum(r['usage'].get('output_tokens',0) for r in x)/1e3,st.mean(r['latency_s'] for r in x),st.mean(r['perf'].get('tool_calls',0) for r in x)))
        if v!='baseline':
            B=load('baseline',s); d=[sc(R[k])-sc(B[k]) for k in R if k in B]
            print('   paired delta %.3f se %.3f'%(st.mean(d),st.stdev(d)/len(d)**.5))
