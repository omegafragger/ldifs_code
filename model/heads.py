import os
import torch
from tqdm import tqdm

import open_clip

from datasets.templates import get_templates
from datasets.registry import get_dataset
from datasets.common import get_dataloader, maybe_dictionarize

from model.modeling import ClassificationHead, ImageEncoder

from sklearn.linear_model import LogisticRegression
from model.logistic_regression import fit_logistic_regression, eval_logistic_regression


def build_zs_classification_head(model, dataset_name, template, data_location, device):
    '''
    This function creates a classification head initializing it using zero-shot weights
    from the text encoder.
    '''

    template = get_templates(dataset_name)
    model = model.model

    logit_scale = model.logit_scale
    dataset = get_dataset(
        dataset_name,
        None,
        location=data_location
    )
    model.eval()
    model.to(device)

    print('Building classification head.')
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(dataset.classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = open_clip.tokenize(texts).to(device) # tokenize
            embeddings = model.encode_text(texts) # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()
        
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights).to(device)

    return classification_head



def get_embeddings(data_loader, model, device):
    embeddings = []
    labels = []

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_loader)):
            batch = maybe_dictionarize(batch)
            images = batch['images'].to(device)
            label = batch['labels'].to(device)
            embedding = model(images)

            embeddings.append(embedding)
            labels.append(label)

        embeddings = torch.cat(embeddings, dim=0)
        # embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        labels = torch.cat(labels, dim=0)

    return embeddings, labels



def linear_probe_pytorch(small_train_embeddings,
                         small_train_labels,
                         val_embeddings,
                         val_labels,
                         train_embeddings,
                         train_labels,
                         device=None):
    '''
    Linear probing function using pytorch which first does a hyperparameter sweep and
    then selects the best model and trains it on the full dataset.
    '''
    lrs = [0.01, 0.05, 0.1, 0.5, 1]

    clf_best = None
    clf_best_acc = 0
    best_lr = lrs[0]

    for lr in lrs:
        clf = fit_logistic_regression(small_train_embeddings, small_train_labels, device=device, lr=lr)
        clf_score = eval_logistic_regression(clf, val_embeddings, val_labels) * 100.
        if (clf_score > clf_best_acc):
            clf_best_acc = clf_score
            best_lr = lr

    clf_best = fit_logistic_regression(train_embeddings, train_labels, device=device, lr=best_lr)
    return clf_best


def linear_probe(small_train_embeddings,
                 small_train_labels,
                 val_embeddings,
                 val_labels,
                 train_embeddings,
                 train_labels):
    '''
    Linear probing function which first does a hyperparameter sweep and then selects
    the best model and trains it on the full dataset.
    '''
    Ls = [1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6]

    clf_best = None
    clf_best_acc = 0
    best_L = Ls[0]

    for L in Ls:
        clf = LogisticRegression(C=1/L, max_iter=1000, verbose=True).fit(small_train_embeddings, small_train_labels)
        clf_score = clf.score(val_embeddings, val_labels) * 100.
        if (clf_score > clf_best_acc):
            clf_best_acc = clf_score
            best_L = L

    clf_best = LogisticRegression(C=1/best_L, max_iter=1000, verbose=True).fit(train_embeddings, train_labels)
    return clf_best



def build_lp_classification_head(model, dataset_name, data_location, device, batch_size=128, pytorch=True):
    '''
    This function creates a classification head initializing it using linear probing
    on the downstream training data.
    '''
    preprocess_fn = model.val_preprocess

    train_val_dataset = get_dataset(
        f'{dataset_name}Val',
        preprocess_fn,
        location=data_location,
    )
    train_test_dataset = get_dataset(
        dataset_name,
        preprocess_fn,
        location=data_location
    )

    small_train_loader = train_val_dataset.train_loader
    val_loader = train_val_dataset.test_loader

    train_loader = train_test_dataset.train_loader


    small_train_embeddings, small_train_labels = get_embeddings(small_train_loader, model, device)
    val_embeddings, val_labels = get_embeddings(val_loader, model, device)
    train_embeddings, train_labels = get_embeddings(train_loader, model, device)

    if pytorch:
        clf_best = linear_probe_pytorch(small_train_embeddings,
                                        small_train_labels,
                                        val_embeddings,
                                        val_labels,
                                        train_embeddings,
                                        train_labels,
                                        device=device)
        return clf_best
    else:
        small_train_embeddings = small_train_embeddings / small_train_embeddings.norm(dim=-1, keepdim=True)
        val_embeddings = val_embeddings / val_embeddings.norm(dim=-1, keepdim=True)
        train_embeddings = train_embeddings / train_embeddings.norm(dim=-1, keepdim=True)

        small_train_embeddings, small_train_labels = small_train_embeddings.cpu().numpy(), small_train_labels.cpu().numpy()
        val_embeddings, val_labels = val_embeddings.cpu().numpy(), val_labels.cpu().numpy()
        train_embeddings, train_labels = train_embeddings.cpu().numpy(), train_labels.cpu().numpy()

        clf_best = linear_probe(small_train_embeddings,
                                small_train_labels,
                                val_embeddings,
                                val_labels,
                                train_embeddings,
                                train_labels)
        linear_weights = torch.tensor(clf_best.coef_).float().to(device)
        linear_bias = torch.tensor(clf_best.intercept_).float().to(device)

        classification_head = ClassificationHead(normalize=True, weights=linear_weights, biases=linear_bias).to(device)
        return classification_head



def get_classification_head(args, dataset, model=None, zs=True):
    filename = os.path.join(args.save, f"{'zs' if zs else 'lp'}_head_{dataset}.pt")
    if model == None or zs:
        model = ImageEncoder(args, keep_lang=True)
    template = get_templates(dataset)

    if (zs):
        classification_head = build_zs_classification_head(model, dataset, template, args.data_location, args.device)
    else:
        classification_head = build_lp_classification_head(model, dataset, args.data_location, args.device)

    os.makedirs(args.save, exist_ok=True)
    classification_head.save(filename)
    return classification_head


def get_classification_head_eval(args, dataset, model=None, zs=True):
    if ((model == None) or zs):
        model = ImageEncoder(args, keep_lang=True)
    template = get_templates(dataset)

    if (zs):
        classification_head = build_zs_classification_head(model, dataset, template, args.data_location, args.device)
    else:
        classification_head = build_lp_classification_head(model, dataset, args.data_location, args.device)

    return classification_head
