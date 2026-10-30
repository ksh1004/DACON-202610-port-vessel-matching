"""CPU compatibility check for real PEFT+optimizer resume checkpoints."""
import argparse
import json

import torch
from peft import PeftModel,get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM,AutoModelForSequenceClassification

from src.common import ROOT
from tools.nested_combiner_experiment import FOLDER,MODELS


def main():
    p=argparse.ArgumentParser();p.add_argument("--kind",choices=["bge","qwen"],default="bge");args=p.parse_args()
    cp=FOLDER/"outer0_inner0"/args.kind/"checkpoint_000200"
    assert (cp/"ready.json").exists()
    state=torch.load(cp/"state.pt",map_location="cpu",weights_only=False)
    torch.set_num_threads(4)
    cls=AutoModelForCausalLM if args.kind=="qwen" else AutoModelForSequenceClassification
    model=cls.from_pretrained(MODELS[args.kind],local_files_only=True,dtype=torch.bfloat16)
    model=PeftModel.from_pretrained(model,cp,is_trainable=True)
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    assert names==state["trainable_names"]
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    optimizer.load_state_dict(state["optimizer"])
    original=load_file(str(cp/"adapter_model.safetensors"))
    loaded=get_peft_model_state_dict(model)
    assert set(original)==set(loaded)
    for k in original:torch.testing.assert_close(original[k],loaded[k],rtol=0,atol=0)
    assert state["completed"]==200 and state["torch_rng"].dtype==torch.uint8
    assert len(state["cuda_rng"])>=1 and all(x.dtype==torch.uint8 for x in state["cuda_rng"])
    for item in optimizer.state.values():
        for k,v in item.items():
            if torch.is_tensor(v):assert torch.isfinite(v).all()
    path=ROOT/"experiments/nested_combiner_checkpoint_checks.json"
    result=json.loads(path.read_text()) if path.exists() else {}
    result[args.kind]={"status":"PASS","step":200,"trainable_parameters_count":len(names),
                       "exact_adapter_restore":True,"optimizer_state_restored":True,"random_states_present":True,
                       "scope":"Actual checkpoint reload compatibility on CPU; not a second full training or exact CUDA continuation comparison"}
    path.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result[args.kind],indent=2),flush=True)


if __name__=="__main__":main()
