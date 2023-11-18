from torch import nn

class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()


    def forward_G(self, d_g_x, real):
        return self.loss(d_g_x, real)
    

    def forward_D(self, d_y, real, d_g_x, fake):
        d_real_loss = self.loss(d_y, real)
        d_fake_loss = self.loss(d_g_x, fake)

        d_loss = (d_real_loss + d_fake_loss)/2

        return d_loss
    


class CycleConsistencyLoss(nn.Module):
    def __init__(self):
        super(CycleConsistencyLoss, self).__init__()
        self.loss_forward = nn.L1Loss()
        self.loss_backward = nn.L1Loss()


    def forward(self, x, y, f_g_x, g_f_y):
        loss_cyc = self.loss_forward(f_g_x, x) + self.loss_backward(g_f_y, y)

        return loss_cyc
    


class IdentityLoss(nn.Module):
    def __init__(self):
        super(IdentityLoss, self).__init__()
        self.loss_x = nn.L1Loss()
        self.loss_y = nn.L1Loss()

    
    def forward(self, x, y, f_y, g_x):
        loss_idt = self.loss_x(f_y, x) + self.loss_y(g_x, y)

        return loss_idt