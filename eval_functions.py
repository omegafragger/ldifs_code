import os
import torch

from datasets.templates import get_templates
from datasets.registry import get_dataset

from metrics.classification_metrics import get_logits_labels, test_classification_net_logits
from metrics.calibration_metrics import ECELoss

from model.modeling import ImageClassifier, ImageEncoder
from model.heads import get_classification_head_eval, build_zs_classification_head, build_lp_classification_head



def get_image_encoder(args, model):
    if args.finetune_loss in ['flyp', 'flyp_ce']:
        image_encoder = model
    else:
        image_encoder = model.image_encoder
    return image_encoder


def evaluate_model(args, model):
    val_preprocess = model.val_preprocess
    eval_dataset = get_dataset(args.eval_dataset, val_preprocess, location=args.data_location, batch_size=128)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_encoder = get_image_encoder(args, model)
    eceloss = ECELoss()

    res_dict = {}
    template = get_templates(args.eval_dataset)

    if args.zs:

        # Create ZS classifier
        zs_classification_head = get_classification_head_eval(args, args.eval_dataset, model=image_encoder, zs=True)
        zs_image_classifier = ImageClassifier(image_encoder, zs_classification_head).to(device)
        zs_logits, zs_labels = get_logits_labels(zs_image_classifier, eval_dataset.test_loader, device)

        # Evaluate ZS accuracy and ECE
        _, zs_acc, _, _, _ = test_classification_net_logits(zs_logits, zs_labels)
        zs_ece = eceloss(zs_logits, zs_labels).item()

        # Storing ZS results
        res_dict['zs_acc'] = zs_acc
        res_dict['zs_ece'] = zs_ece

    if args.lp:

        # Create LP classifier
        lp_classification_head = get_classification_head_eval(args, args.eval_dataset, model=image_encoder, zs=False)
        lp_image_classifier = ImageClassifier(image_encoder, lp_classification_head).to(device)
        lp_logits, lp_labels = get_logits_labels(lp_image_classifier, eval_dataset.test_loader, device)
        
        # Evaluate LP accuracy and ECE
        _, lp_acc, _, _, _ = test_classification_net_logits(lp_logits, lp_labels)
        lp_ece = eceloss(lp_logits, lp_labels).item()
    
        # Storing LP results
        res_dict['lp_acc'] = lp_acc
        res_dict['lp_ece'] = lp_ece

    return res_dict