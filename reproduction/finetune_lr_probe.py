"""One-factor fine-tuning LR comparison from identical pretrained weights and initialization."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('DGLBACKEND','pytorch')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor/cgt'))
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from models import Transformer_class
from util import load_dgl
from ablation_protocol import FixedSamplingGCN, assert_flags
from paper_protocol import degree_statistical_parity


def main():
    out=ROOT/'runs/finetune-lr-probe-seed42'
    out.mkdir(exist_ok=True)
    if (out/'comparison.json').exists():raise SystemExit('Preserve completed comparison.')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    base=ROOT/'runs/cora-baseline-10/seed-42'
    original=json.loads((base/'result.json').read_text())
    data=Planetoid(root=str(ROOT/'data'),name='Cora',split='public')[0]
    saved=torch.load(base/'evaluation.pt',map_location='cpu')
    masks=saved['masks'];degree=saved['degree']
    structure=torch.load(ROOT/'cache'/f"cora-paper-{original['cache_key']}.pt",map_location='cpu')
    b,sim,phi=[structure[k] for k in ['B','sim','phi']];sim=torch.nan_to_num(sim,nan=0.)
    graph=load_dgl(to_networkx(data,to_undirected=True),data.x,sim,phi)
    adj=torch.zeros(data.num_nodes,data.num_nodes,dtype=torch.float64)
    adj[data.edge_index[0],data.edge_index[1]]=1
    checkpoint=base/'pretrained.pt'
    protocol=dict(data_split_seed=42,classifier_initialization_seed=42001,learning_rates=[1e-4,1e-3],
                  finetune_epochs=200,pretrained_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  change='Only fine-tuning learning rate differs between the two new arms; no repeated pretraining.',
                  weight_decay=1e-5,selection='Best validation epoch; one test evaluation after each arm.',
                  initialization_note='Shared new classifier initialization; compare these two arms, not directly with the old seed42 classifier initialization.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    results=[]
    for lr in protocol['learning_rates']:
        torch.manual_seed(protocol['classifier_initialization_seed'])
        model=Transformer_class(data.num_features,64,7,64,4,4,'cora',str(checkpoint),1,1,1)
        with torch.random.fork_rng(devices=[]):
            sampler=FixedSamplingGCN(data.num_features,64,model.model.gcn.anfa,model.model.gcn.beta)
        sampler.load_state_dict(model.model.gcn.state_dict());model.model.gcn=sampler
        assert_flags(model.model,'full')
        initial_hash=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
        optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-5)
        history=[];best_acc=-1.;best_state=None;best_epoch=None;started=time.monotonic()
        device=torch.device('cpu')
        for epoch in range(1,201):
            model.train();optimizer.zero_grad()
            scores=model(graph.local_var(),adj,sim,phi,b,3,epoch,device,device)
            loss=F.cross_entropy(scores[masks['train']],data.y[masks['train']])
            assert torch.isfinite(loss)
            loss.backward();optimizer.step()
            model.eval()
            with torch.no_grad():
                scores=model(graph.local_var(),adj,sim,phi,b,3,epoch,device,device)
                val=float((scores.argmax(1)[masks['validation']]==data.y[masks['validation']]).float().mean())
            if val>best_acc:best_acc,best_epoch,best_state=val,epoch,copy.deepcopy(model.state_dict())
            history.append(dict(epoch=epoch,loss=float(loss),validation_accuracy=val))
            (out/'progress.json').write_text(json.dumps(dict(lr=lr,epoch=epoch,total=200,best_validation_accuracy=best_acc))+'\n')
            if epoch%25==0:print(json.dumps(dict(lr=lr,epoch=epoch,validation_accuracy=val)),flush=True)
        model.load_state_dict(best_state);model.eval()
        rng_before_test=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        with torch.no_grad():pred=model(graph.local_var(),adj,sim,phi,b,3,best_epoch,device,device).argmax(1)
        correct=int((pred[masks['test']]==data.y[masks['test']]).sum())
        fairness=degree_statistical_parity(pred,data.y,degree,masks['test'])
        name=f'lr-{lr:g}'
        torch.save(dict(model=best_state,predictions=pred,labels=data.y,degree=degree,masks=masks),out/f'{name}-evaluation.pt')
        row=dict(lr=lr,initial_state_sha256=initial_hash,rng_before_test_sha256=rng_before_test,
                 best_validation_epoch=best_epoch,best_validation_accuracy=best_acc,
                 accuracy_percent=100*correct/int(masks['test'].sum()),delta_sp_percent=fairness['delta_sp_percent'],
                 history=history,elapsed_seconds=time.monotonic()-started)
        (out/f'{name}-result.json').write_text(json.dumps(row,indent=2)+'\n')
        results.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='history'}),flush=True)
    assert results[0]['initial_state_sha256']==results[1]['initial_state_sha256']
    assert results[0]['rng_before_test_sha256']==results[1]['rng_before_test_sha256']
    result=dict(passed=True,same_initialization_and_RNG_verified=True,arms=[{k:v for k,v in r.items() if k!='history'} for r in results],
                accuracy_difference_points=results[1]['accuracy_percent']-results[0]['accuracy_percent'],
                delta_sp_difference_points=results[1]['delta_sp_percent']-results[0]['delta_sp_percent'],
                scope='One split and one shared initialization; diagnostic only.')
    (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
