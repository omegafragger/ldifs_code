import os
import json
import torch

from args import parse_eval_arguments
from eval_functions import evaluate_model

from datasets.registry import get_dataset
from model.modeling import ImageEncoder

from train_utils import epochs


if __name__ == '__main__':
    args = parse_eval_arguments()
    args.batch_size = 512
    print (args)

    ckp_path = os.path.join(args.model_location, args.model, f'{args.train_dataset}Val')
    if args.finetune_loss in ['ce', 'l2sp', 'ldifs']:
        ckp_path = os.path.join(ckp_path, f"{'zs' if args.zs_init else 'lp'}_init_{args.finetune_loss}{'_fzhd' if args.freeze_head else ''}")
    elif args.finetune_loss in ['flyp', 'flyp_ce']:
        ckp_path = os.path.join(ckp_path, "flyp" if args.finetune_loss == 'flyp' else "flypce")

    # Compute model iteration from index
    model = ImageEncoder(args, keep_lang=False)
    val_preprocess = model.val_preprocess
    train_set = get_dataset(f'{args.train_dataset}Val', val_preprocess, location=args.data_location, batch_size=128)
    num_batches = len(train_set.train_loader)
    train_epochs = epochs[args.train_dataset]
    num_steps = num_batches * train_epochs
    save_every = int(num_steps / 100)
    iteration = args.it_index * save_every

    if args.finetune_loss in ['ce', 'l2sp']:
        model_file_name = f"finetuned_{'zs' if args.zs_init else 'lp'}_init_{args.finetune_loss}_{iteration}{'_fzhd' if args.freeze_head else ''}.pt"
    elif args.finetune_loss == 'ldifs':
        model_file_name = f"finetuned_{'zs' if args.zs_init else 'lp'}_init_{args.finetune_loss}_alpha_{args.ldifs_alpha}_{iteration}{'_fzhd' if args.freeze_head else ''}.pt"
    elif args.finetune_loss in ['flyp', 'flyp_ce']:
        model_file_name = f"finetuned_{args.finetune_loss}_{iteration}.pt"

    model_path = os.path.join(ckp_path, model_file_name)
    model = torch.load(model_path)
    res_dict = evaluate_model(args, model)

    if args.finetune_loss in ['ce', 'l2sp', 'ldifs']:
        res_store_path = os.path.join(args.res_store_path, f'{args.train_dataset}Val', f"{'zs' if args.zs_init else 'lp'}_init_{args.finetune_loss}{'_fzhd' if args.freeze_head else ''}", f'{args.eval_dataset}Val')
    elif args.finetune_loss in ['flyp', 'flyp_ce']:
        res_store_path = os.path.join(args.res_store_path, f'{args.train_dataset}Val', "flyp" if args.finetune_loss == 'flyp' else "flypce", f'{args.eval_dataset}Val')

    if args.finetune_loss in ['ce', 'l2sp']:
        res_save_path = os.path.join(res_store_path, f"res_{args.train_dataset}_{args.eval_dataset}_{'zs' if args.zs_init else 'lp'}_init_{args.finetune_loss}{'_fzhd' if args.freeze_head else ''}_{args.it_index}.json")
    elif args.finetune_loss == 'ldifs':
        res_save_path = os.path.join(res_store_path, f"res_{args.train_dataset}_{args.eval_dataset}_{'zs' if args.zs_init else 'lp'}_init_{args.finetune_loss}_alpha_{args.ldifs_alpha}{'_fzhd' if args.freeze_head else ''}_{args.it_index}.json")
    elif args.finetune_loss in ['flyp', 'flyp_ce']:
        res_save_path = os.path.join(res_store_path, f"res_{args.train_dataset}_{args.eval_dataset}_{args.finetune_loss}_{args.it_index}.json")

    with open(res_save_path, 'w+') as fp:
        json.dump(res_dict, fp)

    print ("Results stored!")