"""ELESON modules and paper equations (11), (14), (16), (24), (26), (30)-(33).

The paper omits ConvLSTM gate geometry and hidden widths. We implement two
stacked 1-D convolutional LSTM cells over non-overlapping temporal frames.
These architectural choices are configurable, not claimed author source code.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, inputs, hidden, kernel_size=3):
        super().__init__()
        self.hidden = hidden
        self.gates = nn.Conv1d(inputs+hidden, 4*hidden, kernel_size, padding=kernel_size//2)

    def forward(self, x, state):
        if state is None:
            h = x.new_zeros(x.shape[0], self.hidden, x.shape[-1])
            c = torch.zeros_like(h)
        else:
            h, c = state
        i, f, o, g = self.gates(torch.cat([x, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f)*c + torch.sigmoid(i)*torch.tanh(g)
        h = torch.sigmoid(o)*torch.tanh(c)
        return h, c


class TemporalEncoder(nn.Module):
    def __init__(self, inputs, outputs, hidden=32, frame_samples=10,
                 kernel_size=3, encoder_width=256, backbone="convlstm"):
        super().__init__()
        self.frame_samples = frame_samples
        self.backbone = backbone
        if backbone == "convlstm":
            self.cell1 = ConvLSTMCell(inputs, hidden, kernel_size)
            self.cell2 = ConvLSTMCell(hidden, hidden, kernel_size)
            size = hidden*frame_samples
        else:
            self.convs = nn.Sequential(nn.Conv1d(inputs, hidden, kernel_size, padding=kernel_size//2),
                nn.ReLU(), nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size//2), nn.ReLU())
            self.lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True)
            size = hidden
        self.head = nn.Sequential(nn.Linear(size, encoder_width), nn.ReLU(),
                                  nn.Linear(encoder_width, outputs), nn.ReLU())

    def forward(self, x):
        b, t, c = x.shape
        if self.backbone == "cnn_lstm":
            sequence, _ = self.lstm(self.convs(x.transpose(1,2)).transpose(1,2))
            return self.head(sequence[:, -1])
        if t % self.frame_samples:
            raise ValueError("window sample count must be divisible by frame_samples")
        frames = x.reshape(b, t//self.frame_samples, self.frame_samples, c).permute(1, 0, 3, 2)
        state1 = state2 = None
        for frame in frames:
            state1 = self.cell1(frame, state1)
            state2 = self.cell2(state1[0], state2)
        return self.head(state2[0].flatten(1))


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return -gradient


class ELESON(nn.Module):
    def __init__(self, samples=200, feature_dim=128, hidden=32,
                 frame_samples=10, noise_std=.05, mode="eleson", features="both",
                 kernel_size=3, encoder_width=256, generator_widths=(256,512),
                 intervention_width=128, adversary_width=64, evidence_widths=(128,64),
                 evidence_bias=1., backbone="convlstm"):
        super().__init__()
        if backbone not in ("convlstm", "cnn_lstm") or kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("backbone must be convlstm/cnn_lstm and kernel_size positive odd")
        if len(generator_widths) != 2 or len(evidence_widths) != 2:
            raise ValueError("generator/evidence require exactly two hidden widths (three FC layers)")
        if min(samples,feature_dim,hidden,frame_samples,encoder_width,intervention_width,
               adversary_width,*generator_widths,*evidence_widths) < 1 or noise_std < 0:
            raise ValueError("network sizes must be positive and noise_std nonnegative")
        self.config = dict(samples=samples, feature_dim=feature_dim, hidden=hidden,
                           frame_samples=frame_samples, noise_std=noise_std, mode=mode, features=features,
                           kernel_size=kernel_size, encoder_width=encoder_width,
                           generator_widths=list(generator_widths), intervention_width=intervention_width,
                           adversary_width=adversary_width, evidence_widths=list(evidence_widths),
                           evidence_bias=evidence_bias, backbone=backbone)
        self.mode, self.noise_std = mode, noise_std
        self.features = features
        self.motion = TemporalEncoder(6, feature_dim*2, hidden, frame_samples, kernel_size, encoder_width, backbone)
        self.magnetic = TemporalEncoder(1, feature_dim, hidden, frame_samples, kernel_size, encoder_width, backbone)
        g1,g2 = generator_widths
        e1,e2 = evidence_widths
        self.generator = nn.Sequential(nn.Linear(feature_dim, g1), nn.ReLU(),
            nn.Linear(g1, g2), nn.ReLU(), nn.Linear(g2, samples*6))
        self.intervention = nn.Sequential(nn.Linear(feature_dim, intervention_width), nn.ReLU(), nn.Linear(intervention_width, 3))
        self.adversary = nn.Sequential(nn.Linear(feature_dim, adversary_width), nn.ReLU(), nn.Linear(adversary_width, 2))
        self.evidence = nn.Sequential(nn.Linear(feature_dim*2, e1), nn.ReLU(),
            nn.Linear(e1, e2), nn.ReLU(), nn.Linear(e2, 3), nn.ReLU())
        # Positive initialization avoids a dead all-zero ReLU evidence collector.
        nn.init.constant_(self.evidence[-2].bias, evidence_bias)
        self.softmax = nn.Linear(feature_dim*2, 3)

    def forward(self, x, auxiliary=False):
        zc, zp = self.motion(x[:, :, :6]).chunk(2, dim=1)
        zb = self.magnetic(x[:, :, 6:])
        z = torch.cat([zc if self.features != "magnetic" else torch.zeros_like(zc),
                       zb if self.features != "motion" else torch.zeros_like(zb)], dim=1)
        if self.mode == "e2e":
            logits = self.softmax(z)
            prob = logits.softmax(-1)
            out = dict(confidence=prob, probability=prob, uncertainty=prob.new_zeros(len(x)), logits=logits)
        else:
            e = self.evidence(z)
            strength = e.sum(-1, keepdim=True)
            # Eq.26 is belief mass e/(sum(e)+3), NOT Dirichlet mean (e+1)/S.
            out = dict(evidence=e, confidence=e/(strength+3),
                       probability=(e+1e-8)/(strength+3e-8),
                       uncertainty=3/(strength.squeeze(-1)+3))
        if auxiliary:
            noise = torch.randn_like(zc)*self.noise_std if self.training else torch.zeros_like(zc)
            out.update(zc=zc, zp=zp, zb=zb,
                       reconstruction=self.generator(zc+zp+noise).reshape(len(x), -1, 6),
                       combined_logits=self.intervention(zc+zp),
                       control_logits=self.intervention(zp),
                       behavior_logits=self.adversary(ReverseGradient.apply(zb)))
        return out


def objective(out, x, y, behavior, *, w1=.6, w2=.3, w3=.4, w4=1.,
              causal=True, adversarial=True, uncertainty=True,
              con_reduction="sample", adv_reduction="batch", variance_weight=1.):
    if "logits" in out:
        loss = F.cross_entropy(out["logits"], y)
        return loss, {"classification": loss.detach(), "optimization_loss": loss.detach()}
    e = out["evidence"]
    cls = F.nll_loss(out["probability"].clamp_min(1e-12).log(), y)
    alpha = e+1
    strength = alpha.sum(-1, keepdim=True)
    variance = (alpha*(strength-alpha)/(strength.square()*(strength+1))).sum(-1).mean()
    rec = F.mse_loss(out["reconstruction"], x[:, :, :6])
    sim = F.cross_entropy(out["combined_logits"], y) + F.cross_entropy(out["control_logits"], torch.zeros_like(y))
    if con_reduction not in ("sample", "class") or adv_reduction not in ("batch", "conveyor"):
        raise ValueError("invalid loss reduction")
    terms = [out["zc"][y == c].var(dim=0, unbiased=False).mean() *
             ((y == c).sum()/len(y) if con_reduction == "sample" else 1.) for c in y.unique()
             if con_reduction == "sample" or (y == c).sum() > 1]
    con = ((torch.stack(terms).sum() if con_reduction == "sample" else torch.stack(terms).mean())
           if terms else out["zc"].sum()*0)
    conveyor = y != 0  # The binary s=1 notation covers BOTH elevator and escalator.
    adv = (F.cross_entropy(out["behavior_logits"][conveyor], behavior[conveyor])
           if conveyor.any() else out["behavior_logits"].sum()*0)
    if adv_reduction == "batch":
        adv = adv * conveyor.float().mean()
    cal = sim+w1*rec+w2*con
    els = cls + (variance_weight*variance if uncertainty else 0)
    # GRL reverses only the feature gradient. The adversary minimizes CE while
    # the magnetic encoder maximizes it. Negating this scalar would reverse
    # the adversary's own gradient, an incorrect implementation of Eq.24.
    loss = els + (w4*cal if causal else 0) + (w3*adv if adversarial else 0)
    values = dict(classification=cls, dirichlet_variance=variance, reconstruction=rec,
                  simulation=sim, constraint=con, behavior_ce=adv,
                  causal=cal, optimization_loss=loss)
    # This value matches Eq.33 for the encoder; optimization_loss includes the
    # positive discriminator CE used with GRL and is not Eq.33's scalar value.
    values["encoder_objective"] = els + (w4*cal if causal else 0) - (w3*adv if adversarial else 0)
    return loss, {k:v.detach() for k,v in values.items()}
