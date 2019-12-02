import torch
from torch.functional import F
from tqdm.autonotebook import tqdm


def train_epoch(model, dataset, device, *, reg_weight, learning_rate, batch_size,
                loss_print_freq=500, max_batches=None):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    with tqdm(enumerate(loader), total=len(dataset)/batch_size, desc='Batch') as pipe:
        for batch_idx, batch in pipe:
            for k, v in batch.items():
                batch[k] = v.to(device)
            y, (v_a, _), (v_b, m_b) = model(batch['frame_a'],
                                            batch['frame_b'],
                                            batch['amplification_f'])
            v_c, m_c = model.encoder(batch['frame_perturbed'])
            loss = F.l1_loss(y, batch['frame_amplified']) + \
                reg_weight * F.l1_loss(v_a, v_b) + \
                reg_weight * F.l1_loss(v_a, v_c) + \
                reg_weight * F.l1_loss(m_b, m_c)
            if batch_idx % loss_print_freq == 0:
                pipe.write("Batch %d loss %.2f" % (batch_idx+1, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if max_batches is not None and batch_idx + 1 >= max_batches:
                break
