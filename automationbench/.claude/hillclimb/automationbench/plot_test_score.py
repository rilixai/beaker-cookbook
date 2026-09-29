import json,glob,statistics as st
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
def L(v,s): return [json.load(open(f)) for f in glob.glob(f'{v}/{s}/*.json') if not f.endswith(('config.json','summary.json'))]
sc=lambda r:.8*r['partial_credit']+.2*r['task_completed_correctly']
V=['baseline','v1','v2']; X=[0,1,2]
def stat(s):
    m=[];e=[]
    for v in V:
        x=[sc(r) for r in L(v,s)]; m.append(st.mean(x)); e.append(st.stdev(x)/len(x)**.5)
    return m,e
tm,te=stat('test'); rm,_=stat('train')
fig,ax=plt.subplots(figsize=(7,4.2),dpi=160)
ax.errorbar(X,tm,yerr=te,color='#2a6fdb',marker='o',ms=8,lw=2,capsize=4,label='Test (18 tasks, ±1 SE)',zorder=3)
ax.plot(X,rm,color='#8a8f98',marker='o',ms=6,lw=1.5,ls='--',label='Train (36 tasks)',zorder=2)
for x,y in zip(X,tm): ax.annotate(f'{y:.3f}',(x,y),textcoords='offset points',xytext=(0,-22),ha='center',fontsize=10,color='#1f2937')
ax.set_xticks(X); ax.set_xticklabels(['0\nbaseline','1\npolicy sweep\n(kept)','2\nscreening\n(reverted)'],fontsize=9)
ax.set_ylim(0.55,0.88); ax.set_ylabel('score = 0.8·partial + 0.2·pass',fontsize=9)
ax.grid(axis='y',color='#e5e7eb',lw=.8); ax.set_axisbelow(True)
for s in['top','right']: ax.spines[s].set_visible(False)
ax.legend(frameon=False,fontsize=9,loc='upper right'); ax.set_title('AutomationBench hillclimb: test score by round',fontsize=11,loc='left')
plt.tight_layout(); plt.savefig('test_score_by_round.png'); print(tm,te,rm)
