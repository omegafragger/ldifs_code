import os
import time

import torch
import open_clip
import torch.nn.functional as F

from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from datasets.templates import get_templates

from model.modeling import ImageEncoder, ImageEncoderAugmented, ImageClassifier
from model.heads import get_classification_head

from train_utils import cosine_lr

from loss.l2_sp import L2SP
from loss.ldifs import CE_LDIFS
from loss.clip_loss import ClipLoss
from loss.label_smoothing import LabelSmoothing

from metrics.classification_metrics import test_classification_net


def get_best_pretrained_encoder(args, image_encoder_list, trainval_dataset):
    '''
    Get the best pre-trained encoder to be used for training on the next dataset.

    args: training args
    image_encoder_list: list of image encoders per task on which fine-tuning has already been done
    trainval_dataset: train val dataset
    '''
    val_loader = trainval_dataset.test_loader # This is the trainVal dataset
    
    max_acc = 0
    best_index = -1
    if (args.best_among_list):
        for i, image_encoder in enumerate(image_encoder_list):
            classification_head = get_classification_head(args, args.train_dataset, model=image_encoder, zs=args.zs_init).to(args.device)
            image_classifier = ImageClassifier(image_encoder, classification_head).to(args.device)
            _, acc, _, _, _ = test_classification_net(image_classifier, val_loader, args.device)
            
            if (acc > max_acc):
                max_acc = acc
                best_index = i
    best_model = ImageEncoderAugmented(args, keep_lang=False).to(args.device)
    best_model.load_state_dict(image_encoder_list[best_index].state_dict())

    if (args.weight_interpolate):
        pretrained_model = ImageEncoderAugmented(args, keep_lang=False).to(args.device)
        weight_alphas = [0, 0.25, 0.5, 0.75, 1.]

        max_acc = 0
        best_weight_alpha = weight_alphas[0]
        for weight_alpha in weight_alphas:
            cur_model = ImageEncoderAugmented(args, keep_lang=False).to(args.device)
            for name, param in pretrained_model.named_parameters():
                cur_model.state_dict()[name] = (weight_alpha * param) + ((1 - weight_alpha) * best_model.state_dict()[name])
            classification_head = get_classification_head(args, args.train_dataset, model=cur_model, zs=args.zs_init).to(args.device)
            cur_classifier = ImageClassifier(cur_model, classification_head).to(args.device)
            _, acc, _, _, _ = test_classification_net(cur_classifier, val_loader, args.device)

            if (acc > max_acc):
                mac_acc = acc
                best_weight_alpha = weight_alpha
        cur_model = ImageEncoderAugmented(args, keep_lang=False).to(args.device)
        for name, param in pretrained_model.named_parameters():
            cur_model.state_dict()[name] = (best_weight_alpha * param) + ((1 - best_weight_alpha) * best_model.state_dict()[name])
        best_model = cur_model

    return best_model


def finetune_ldifs(args, finetuned_image_encoder=[]):
    '''
    Method to fine-tune using the LDIFS loss.
    '''
    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, args.model, args.train_dataset, f"{'zs' if args.zs_init else 'lp'}_init_{args.finetune_loss}{'_fzhd' if args.freeze_head else ''}")

    assert train_dataset is not None, "Please provide a training dataset."
    print('Building image encoder.')

    if (len(finetuned_image_encoder) == 0):
        image_encoder = ImageEncoderAugmented(args, keep_lang=False).to(args.device)
    else:
        image_encoder = finetuned_image_encoder[-1]
        image_encoder = image_encoder.to(args.device)
    classification_head = get_classification_head(args, train_dataset, model=image_encoder, zs=args.zs_init).to(args.device)

    model = ImageClassifier(image_encoder, classification_head)
    if args.freeze_head:
        model.freeze_head()

    preprocess_fn = model.train_preprocess
    print_every = 100

    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
        image_text=False
    )
    num_batches = len(dataset.train_loader)
    num_steps = num_batches * args.epochs
    save_every = int(num_steps / 100)

    pretrained_image_encoder = ImageEncoderAugmented(args, keep_lang=False).cuda()
    if (len(finetuned_image_encoder) > 0):
        pretrained_image_encoder = get_best_pretrained_encoder(args, finetuned_image_encoder, dataset)
        pretrained_image_encoder = pretrained_image_encoder.to(args.device)
    loss_fn = CE_LDIFS(pretrained_image_encoder, model, alpha=args.ldifs_alpha)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    # Saving zero-shot model
    if args.save is not None:
        os.makedirs(ckpdir, exist_ok=True)
        model_path = os.path.join(ckpdir, f"zeroshot_{'zs' if args.zs_init else 'lp'}_init.pt")
        model.save(model_path)

    for epoch in range(args.epochs):
        model = model.cuda()
        model.train()
        data_loader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=None)

        for i, batch in enumerate(data_loader):
            start_time = time.time()
            
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch['images'].to(args.device)
            labels = batch['labels'].to(args.device)
            data_time = time.time() - start_time

            logits = model(inputs)
            loss = loss_fn(inputs, logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            batch_time = time.time() - start_time

            if step % print_every == 0:
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )
            if step % save_every == 0:
                ft_path = os.path.join(ckpdir, f"finetuned_{'zs' if args.zs_init else 'lp'}_init_{args.finetune_loss}_alpha_{args.ldifs_alpha}_{step}{'_fzhd' if args.freeze_head else ''}.pt")
                # model.module.save(ft_path)
                model.save(ft_path)
                model = model.cuda()

    if args.save is not None:
        zs_path = os.path.join(ckpdir, f"zeroshot_{'zs' if args.zs_init else 'lp'}_init{'_fzhd' if args.freeze_head else ''}.pt")
        ft_path = os.path.join(ckpdir, f"finetuned_{'zs' if args.zs_init else 'lp'}_init_{args.finetune_loss}_alpha_{args.ldifs_alpha}{'_fzhd' if args.freeze_head else ''}.pt")
        model.save(ft_path)
        print ('Finetune complete!!')
        return model

    return model



def finetune_ft(args, finetuned_image_encoder=[]):
    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, args.model, args.train_dataset, f"{'zs' if args.zs_init else 'lp'}_init_{args.finetune_loss}{'_fzhd' if args.freeze_head else ''}")

    assert train_dataset is not None, "Please provide a training dataset."
    print('Building image encoder.')
    if (len(finetuned_image_encoder) == 0): 
        image_encoder = ImageEncoder(args, keep_lang=False).to(args.device)
    else:
        image_encoder = finetuned_image_encoder[-1]
        image_encoder = image_encoder.to(args.device)
    classification_head = get_classification_head(args, train_dataset, model=image_encoder, zs=args.zs_init).to(args.device)

    model = ImageClassifier(image_encoder, classification_head)
    if args.freeze_head:
        model.freeze_head()

    preprocess_fn = model.train_preprocess
    print_every = 100

    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
        image_text=False
    )
    num_batches = len(dataset.train_loader)
    num_steps = num_batches * args.epochs
    save_every = int(num_steps / 100)

    if args.finetune_loss == 'ce':
        loss_fn = torch.nn.CrossEntropyLoss()
    elif args.finetune_loss == 'ls':
        loss_fn = LabelSmoothing(args.ls_factor)
    elif args.finetune_loss == 'l2sp':
        pretrained_image_encoder = ImageEncoder(args, keep_lang=False).cuda()
        if len(finetuned_image_encoder) > 0:
            pretrained_image_encoder.load_state_dict(finetuned_image_encoder[-1].state_dict())
            pretrained_image_encoder = pretrained_image_encoder.to(args.device)
        loss_fn = L2SP(pretrained_image_encoder, model, alpha=args.l2sp_alpha)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    # Saving zero-shot model
    if args.save is not None:
        os.makedirs(ckpdir, exist_ok=True)
        model_path = os.path.join(ckpdir, f"zeroshot_{'zs' if args.zs_init else 'lp'}_init.pt")
        # model.module.image_encoder.save(model_path)
        model.save(model_path)

    for epoch in range(args.epochs):
        model = model.cuda()
        model.train()
        data_loader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=None)

        for i, batch in enumerate(data_loader):
            start_time = time.time()
            
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch['images'].to(args.device)
            labels = batch['labels'].to(args.device)
            data_time = time.time() - start_time

            logits = model(inputs)
            loss = loss_fn(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            batch_time = time.time() - start_time

            if step % print_every == 0:
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )
            if step % save_every == 0:
                ft_path = os.path.join(ckpdir, f"finetuned_{'zs' if args.zs_init else 'lp'}_init_{args.finetune_loss}_{step}{'_fzhd' if args.freeze_head else ''}.pt")
                # model.module.save(ft_path)
                model.save(ft_path)
                model = model.cuda()


    if args.save is not None:
        zs_path = os.path.join(ckpdir, f"zeroshot_{'zs' if args.zs_init else 'lp'}_init{'_fzhd' if args.freeze_head else ''}.pt")
        ft_path = os.path.join(ckpdir, f"finetuned_{'zs' if args.zs_init else 'lp'}_init_{args.finetune_loss}{'_fzhd' if args.freeze_head else ''}.pt")
        # model.module.save(ft_path)
        model.save(ft_path)
        print ('Finetune complete!!')
        return model

    return model



def finetune_flyp(args, finetuned_clip_encoder=[]):
    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, args.model, args.train_dataset, "flyp")

    assert train_dataset is not None, "Please provide a training dataset."
    print('Building image encoder.')

    if (len(finetuned_clip_encoder) == 0):
        clip_encoder = ImageEncoder(args, keep_lang=True).to(args.device)
    else:
        clip_encoder = finetuned_clip_encoder[-1]
        clip_encoder = clip_encoder.to(args.device)

    model = clip_encoder

    preprocess_fn = model.train_preprocess
    print_every = 100

    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
        image_text=True
    )
    num_batches = len(dataset.train_loader)
    num_steps = num_batches * args.epochs
    save_every = int(num_steps / 100)

    model = model.cuda()
    model.train()

    clip_loss_fn = ClipLoss(local_loss=False,
                            gather_with_grad=False,
                            cache_labels=True,
                            rank=0,
                            world_size=1,
                            use_horovod=False)

    clip_params = list(model.parameters())
    total_params = clip_params
    params = [p for p in total_params if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    # Saving zero-shot model
    if args.save is not None:
        os.makedirs(ckpdir, exist_ok=True)
        model_path = os.path.join(ckpdir, f"zeroshot_flyp.pt")
        # model.module.image_encoder.save(model_path)
        # model.module.save(model_path)
        model.save(model_path)

    stats = []
    for epoch in range(0, args.epochs):
        print ("Epoch: ", epoch)
        epoch_stats = {}
        epoch_stats['epoch'] = epoch
        id_flyp_loss_sum = 0
        model.cuda()
        model.train()
        model = model.train()
        # classification_head.train()

        data_loader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=None)

        for i, batch in enumerate(data_loader):
            start_time = time.time()
            
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch['images'].to(args.device)
            labels = batch['labels'].to(args.device)
            data_time = time.time() - start_time

            image_features, text_features, logit_scale2 = model(inputs, labels)
            clip_loss = clip_loss_fn(image_features,
                                     text_features,
                                     logit_scale2.item())
            clip_loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            id_flyp_loss_sum += clip_loss.item()
            batch_time = time.time() - start_time

            if step % print_every == 0:
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Loss: {id_flyp_loss_sum:.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )
            if step % save_every == 0:
                ft_path = os.path.join(ckpdir, f'finetuned_{args.finetune_loss}_{step}.pt')
                # model.module.save(ft_path)
                model.save(ft_path)
                model = model.cuda()
                model.train()

    if args.save is not None:
        zs_path = os.path.join(ckpdir, f'zeroshot_flyp.pt')  
        ft_path = os.path.join(ckpdir, f'finetuned_{args.finetune_loss}.pt')
        # model.module.save(ft_path)
        model.save(ft_path)
        print ('Finetune complete!!')
        return model

    return model



def finetune_flyp_ce(args, finetuned_clip_encoder=[]):
    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, args.model, args.train_dataset, "flypce")

    assert train_dataset is not None, "Please provide a training dataset."
    print('Building image encoder.')

    if (len(finetuned_clip_encoder) == 0):
        clip_encoder = ImageEncoder(args, keep_lang=True).to(args.device)
    else:
        clip_encoder = finetuned_clip_encoder[-1]
        clip_encoder = clip_encoder.to(args.device)

    model = clip_encoder

    preprocess_fn = model.train_preprocess
    print_every = 100

    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
        image_text=False
    )
    template = get_templates(args.train_dataset)

    num_batches = len(dataset.train_loader)
    num_steps = num_batches * args.epochs
    save_every = int(num_steps / 100)

    model = model.cuda()
    model.train()

    clip_params = list(model.parameters())
    total_params = clip_params
    params = [p for p in total_params if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    # Saving zero-shot model
    if args.save is not None:
        os.makedirs(ckpdir, exist_ok=True)
        model_path = os.path.join(ckpdir, f"zeroshot_flyp_ce.pt")
        model.save(model_path)

    # Code for CE Ablation
    all_texts = []
    for classname in dataset.classnames:
        texts = []
        for t in template:
            texts.append(t(classname))
        texts = open_clip.tokenize(texts)
        all_texts.append(texts)

    all_texts = torch.stack(all_texts, dim=0)

    assert all_texts.shape[0] == len(dataset.classnames)
    assert all_texts.shape[1] == len(template)
    assert all_texts.shape[2] == 77

    stats = []
    for epoch in range(0, args.epochs):
        print ("Epoch: ", epoch)
        epoch_stats = {}
        epoch_stats['epoch'] = epoch
        id_ce_loss_sum = 0
        model = model.cuda()
        model.train()
        model = model.train()

        data_loader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=None)

        for i, batch in enumerate(data_loader):
            start_time = time.time()
            
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch['images'].to(args.device)
            labels = batch['labels'].to(args.device)
            data_time = time.time() - start_time

            # Sample prompts for #C classes
            b = torch.arange(len(dataset.classnames))
            s = torch.randint(low=0,
                              high=all_texts.shape[1],
                              size=(all_texts.shape[0], ))
            current_texts = all_texts[b, s, :]
            current_texts = current_texts.cuda()

            assert current_texts.shape[0] == len(dataset.classnames)
            assert current_texts.shape[1] == 77

            image_features = model(inputs, None)
            text_features = model(None, current_texts)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = model.model.logit_scale.exp()

            assert text_features.shape[0] == len(dataset.classnames)
            logits = logit_scale * image_features @ text_features.T
            xent_loss = F.cross_entropy(logits, labels)

            xent_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            id_ce_loss_sum += xent_loss.item()
            batch_time = time.time() - start_time

            if step % print_every == 0:
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Loss: {id_ce_loss_sum:.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )
            if step % save_every == 0:
                ft_path = os.path.join(ckpdir, f'finetuned_{args.finetune_loss}_{step}.pt')
                # model.module.save(ft_path)
                model.save(ft_path)
                model = model.cuda()
                model.train()

    if args.save is not None:
        zs_path = os.path.join(ckpdir, f'zeroshot_flyp_ce.pt')  
        ft_path = os.path.join(ckpdir, f'finetuned_{args.finetune_loss}.pt')
        #model.module.save(ft_path)
        model.save(ft_path)
        print ('Finetune complete!!')
        return model

    return model
