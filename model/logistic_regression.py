import torch

from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from model.modeling import ClassificationHead


def fit_logistic_regression(train_embeddings, train_labels, device, max_epochs=1000, lr=1.0):
    '''
    Fit a logistic regression model using embeddings and labels
    train_embeddings: [B, D] shaped tensor
    train_labels: [B] shaped tensor
    device: Pytorch device
    max_epochs: max epochs for optimization
    lr: learning rate for optimizer
    '''
    num_labels = torch.unique(train_labels).shape[0]
    input_dim = train_embeddings.shape[1]
    pseudo_lin_layer = nn.Linear(input_dim, num_labels)
    logistic_regression_model = ClassificationHead(normalize=True,
                                                   weights=pseudo_lin_layer.weight,
                                                   biases=pseudo_lin_layer.bias).to(device)
    logistic_regression_model.train()

    optimizer = torch.optim.AdamW(params=logistic_regression_model.parameters(), lr=lr)

    for it in tqdm(range(max_epochs)):
        optimizer.zero_grad()
        logits = logistic_regression_model(train_embeddings)
        loss = F.cross_entropy(logits, train_labels)
        loss.backward()
        optimizer.step()

    return logistic_regression_model


def eval_logistic_regression(logistic_regression_model, embeddings, labels):
    '''
    Evaluate the accuracy of a logistic regression model using embeddings and labels.

    logistic_regression_model: model to be evaluated
    embeddings: eval embeddings
    labels: eval labels
    '''
    logistic_regression_model.eval()
    with torch.no_grad():
        pred_logits = logistic_regression_model(embeddings)
        pred_labels = torch.argmax(pred_logits, dim=-1).squeeze()
        return ((pred_labels == labels).sum() / float(labels.shape[0]))
