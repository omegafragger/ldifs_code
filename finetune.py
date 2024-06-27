import torch
from args import parse_finetune_arguments
from finetune_functions import finetune_ft, finetune_ldifs, finetune_flyp, finetune_flyp_ce
from train_utils import epochs


def finetune(args, finetuned_encoder=[]):

    if (args.finetune_loss in ['ce', 'ls', 'l2sp']):
        finetuned_image_classifier = finetune_ft(args, finetuned_image_encoder=finetuned_encoder)
        return finetuned_image_classifier

    elif (args.finetune_loss == 'ldifs'):
        finetuned_image_classifier = finetune_ldifs(args, finetuned_image_encoder=finetuned_encoder)
        return finetuned_image_classifier

    elif (args.finetune_loss == 'flyp'):
        finetuned_clip_encoder = finetune_flyp(args, finetuned_clip_encoder=finetuned_encoder)
        return finetuned_clip_encoder

    elif (args.finetune_loss == 'flyp_ce'):
        finetuned_clip_encoder = finetune_flyp_ce(args, finetuned_clip_encoder=finetuned_encoder)
        return finetuned_clip_encoder



if __name__ == '__main__':

    args = parse_finetune_arguments()
    args.lr = 1e-5

    image_encoders = []
    for i, train_dataset in enumerate(args.train_datasets):
        print('='*100)
        print(f'Finetuning {args.model} on {train_dataset}')
        print('='*100)
        args.epochs = epochs[train_dataset]
        args.train_dataset = train_dataset + 'Val'
        #args.batch_size = 128

        if (i == 0 and args.model_checkpoint_path is not None):
            finetuned_checkpoint = torch.load(args.model_checkpoint_path)
            if (args.finetune_loss in ['ce', 'ls', 'l2sp', 'ldifs']):
                image_encoders.append(finetuned_checkpoint.image_encoder)
            elif (args.finetune_loss in ['flyp', 'flyp_ce']):
                image_encoders.append([finetuned_checkpoint])
            print ('Checkpoint loaded!')
        elif (i == 0 and args.model_checkpoint_path is None):
            print ('No finetuned checkpoint')
        elif (i > 0):
            if (args.finetune_loss in ['ce', 'ls', 'l2sp', 'ldifs']):
                image_encoders.append(finetuned_model.image_encoder)
            elif (args.finetune_loss in ['flyp', 'flyp_ce']):
                image_encoders.append(finetuned_model)

        print (args)
        finetuned_model = finetune(args, finetuned_encoder=image_encoders)