import torch
import torch.nn as nn

def distance_matrix(x, y):

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


class KNN(nn.Module):
    def __init__(self, X=None, Y=None, k=2):
        super(KNN, self).__init__()
        self.k = k
        self.train(X, Y)

    def train(self, X=None, Y=None):
        self.train_pts = X
        self.train_label = Y
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def forward(self, x):
        dist = distance_matrix(x, self.train_pts, 2) ** 0.5

        knn = dist.topk(self.k, largest=False)
        votes = self.train_label[knn.indices]

        winner = torch.zeros(votes.size(0), dtype=votes.dtype, device=votes.device)
        count = torch.zeros(votes.size(0), dtype=votes.dtype, device=votes.device) - 1

        for lab in self.unique_labels:
            vote_count = (votes == lab).sum(1)
            who = vote_count >= count
            winner[who] = lab
            count[who] = vote_count[who]

        return winner