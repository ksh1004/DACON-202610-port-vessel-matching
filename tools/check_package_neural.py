"""Exercise real neural calls in an isolated packaged-script process."""
import argparse
import subprocess
import sys
from pathlib import Path


CODE = r'''
import gc,json,sys,tempfile
from pathlib import Path
import pandas as pd
import torch
import script
from src.neural_pair import PairScorer

calls=[]
view_calls={}
if (script.MODEL_DIR/'encoder_views.json').exists():
    from src.encoder_views import ViewScorer
    package_views=json.loads((script.MODEL_DIR/'encoder_views.json').read_text())['views']
    signatures=[v['kind']+':'+v['recipe'] for v in package_views]
    original=ViewScorer.score
    def counted(self,row,candidates,beams):
        calls.append(row['sample_id'])
        label=self.name
        if signatures.count(label)>1:
            label+=':'+Path(self.scorer.model.config._name_or_path).name
        view_calls[label]=view_calls.get(label,0)+1
        return original(self,row,candidates,beams)
    ViewScorer.score=counted
elif (script.MODEL_DIR/'pair_views.json').exists():
    from src.pair_views import PairViewScorer
    original=PairViewScorer.score
    def counted(self,row,candidates,beams=()):
        calls.append(row['sample_id'])
        view_calls[self.recipe]=view_calls.get(self.recipe,0)+1
        return original(self,row,candidates,beams)
    PairViewScorer.score=counted
else:
    original=PairScorer.score
    def counted(self,row,candidates):
        calls.append(row['sample_id'])
        return original(self,row,candidates)
    PairScorer.score=counted
data=pd.read_csv(sys.argv[1],encoding='utf-8-sig').drop(columns=['target_vessel_id'],errors='ignore')
catalog=Path(sys.argv[2]).resolve()
with tempfile.TemporaryDirectory(prefix='neural_row_check_') as temp:
    folder=Path(temp)
    def run(frame,label):
        source=folder/(label+'_input.csv');target=folder/(label+'_output.csv')
        frame.to_csv(source,index=False,encoding='utf-8')
        script.run(source,catalog,target)
        result=pd.read_csv(target,dtype=str)
        gc.collect();torch.cuda.empty_cache()
        return dict(zip(result.sample_id,result.target_vessel_id))
    full=run(data,'full')
    full_calls=list(calls)
    full_view_calls=dict(view_calls)
    assert full_calls,'Stress cases did not exercise the neural scoring branch'
    if (script.MODEL_DIR/'encoder_views.json').exists():
        expected={v['kind']+':'+v['recipe']+(':'+v['dir'] if signatures.count(v['kind']+':'+v['recipe'])>1 else '') for v in package_views}
        assert set(full_view_calls)==expected
        assert all(count==len(set(full_calls)) for count in full_view_calls.values())
    elif (script.MODEL_DIR/'pair_views.json').exists():
        expected={v['recipe'] for v in json.loads((script.MODEL_DIR/'pair_views.json').read_text())['views']}
        assert set(full_view_calls)==expected
        assert all(count==len(set(full_calls)) for count in full_view_calls.values())
    shuffled=run(data.sample(frac=1,random_state=42),'shuffled')
    assert full==shuffled
    singles=list(dict.fromkeys(full_calls))[:2]
    for j,sample in enumerate(singles):
        one=run(data.loc[data.sample_id.eq(sample)],'single'+str(j))
        assert one[sample]==full[sample]
    print(json.dumps({'status':'PASS','neural_calls_full':len(full_calls),'active_neural_rows_full':len(set(full_calls)),'view_calls_full':full_view_calls,'targeted_neural_singles':len(singles),'shuffled_rows':len(data),'view_calls_all_runs':view_calls}))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--cases", type=Path, default=Path("experiments/neural_independence_cases.csv"))
    parser.add_argument("--catalog", type=Path, default=Path("vessel_catalog.csv"))
    args = parser.parse_args()
    result = subprocess.run([sys.executable, "-c", CODE, str(args.cases.resolve()), str(args.catalog.resolve())],
                            cwd=args.package.resolve(), timeout=1200)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
