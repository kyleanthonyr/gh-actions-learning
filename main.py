import torch.nn as nn


class Net(nn.Module):
    """
    Multi-task learning model which uses some shared layers,
    and task-specific segments.
    """

    def __init__(self, h):
        super(Net, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
        )

        self.model_sin = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )

        self.model_cos = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )

    def forward(self, inputs):
        # pass through shared layers
        x1 = self.model(inputs)

        # generate sin(x) prediction
        output_sin = self.model_sin(x1)

        # generate cos(x) prediction
        output_cos = self.model_cos(x1)

        return output_sin, output_cos


if __name__ == "__main__":
    net = Net(150)
    loss_func = nn.MSELoss()

    epochs = 50

    for epoch in range(epochs):
        sin_pred, cos_pred = net(x)

        # compute loss
        loss1 = loss_func(sin_pred, sin_true)
        loss2 = loss_func(cos_pred, cos_true)

        # add loss
        loss = loss1 + loss2

        # run backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
