import torch

def nt_xent(z1, z2, tau=0.2):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = (z @ z.T) / tau
    mask = torch.eye(2*B, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)
    pos = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)])
    denom = torch.logsumexp(sim, dim=1)
    return -(pos - denom).mean()
