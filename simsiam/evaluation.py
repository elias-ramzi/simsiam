import torch
from tqdm import tqdm

from .knn import FaissKNeighbors


def compute_embeddings(net, loader, with_labels=True, to_cpu=False, to_numpy=False, verbose=True):
    if to_numpy:
        to_cpu = True

    train_features = []
    train_labels = []
    iterator = tqdm(loader) if verbose else loader
    for (img, lb) in iterator:
        with torch.no_grad():
            di = net(img.cuda())
        if to_cpu:
            train_features.append(di.cpu())
        else:
            train_features.append(di)
        train_labels.append(lb)

    X = torch.cat(train_features)
    y = torch.cat(train_labels)

    if to_numpy:
        return X.numpy(), y.numpy()
    return X, y


def knn_evaluation(net, train_loader, test_loader, k=20, verbose=True):
    X_train, y_train = compute_embeddings(net, train_loader, to_numpy=True, verbose=verbose)
    X_test, y_test = compute_embeddings(net, test_loader, to_numpy=True, verbose=verbose)

    knn_classifier = FaissKNeighbors(k=k)
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)

    return (y_test == y_pred).mean()
