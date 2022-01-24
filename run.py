import torch
from torch import dtype, nn, optim
import datetime
from dataloader import DataLoader, Tokenizer
from model import Backbone, Transformer
from utils import calc_batch_accuracy, write_metric
import config as cfg
from torch.optim.lr_scheduler import ReduceLROnPlateau


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the vocabulary
vocab_file = open(cfg.VOCABULARY_FILE, 'r')
idx2word = vocab_file.read().splitlines()
vocab_size = len(idx2word)
word2idx = {word:idx for idx, word in enumerate(idx2word)}
vocab_file.close()

# Intialize DataLoader and Tokenizer
loader = DataLoader(videos_path=cfg.VIDEOS_DIR, annotations_path=cfg.ANNOTATIONS_DIR,
 initial_batch_size=cfg.INITIAL_BATCH_SIZE, shuffle=True)
tokenizer = Tokenizer(word2idx)

# Initialize the backbone and the transformer models
backbone = Backbone()
backbone = backbone.to(device)
transformer = Transformer(vocab_size)
transformer = transformer.to(device)

# Initialize optimizers and loss function
backbone_optimizer = optim.Adam(backbone.parameters())
transformer_optimizer = optim.SGD(transformer.parameters(), lr=1e-1)
loss_fn=nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])

backbone_cheduler = ReduceLROnPlateau(backbone_optimizer, mode='min', factor=0.1, patience=2, verbose=True)
transformer_cheduler = ReduceLROnPlateau(transformer_optimizer, mode='min', factor=0.1, patience=2, verbose=True)


losses = []
all_accs = []
all_loss = []

# Training loop
backbone.train()
transformer.train()
#difficulty_increases = 0
for epoch in range(cfg.EPOCHS):
    print(f'<--------------------- Epoch: {epoch + 1} --------------------->')
    epoch_accs = []
    tot_loss = 0
    count = 0
    for samples, labels in loader:
        t1 = datetime.datetime.now()
        tokenizer_res = tokenizer.tokenize(samples, labels)
        batch_inputs, batch_targets,\
        batch_in_pad_masks, batch_tgt_pad_masks = tokenizer_res

        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        batch_in_pad_masks = batch_in_pad_masks.to(device)
        batch_tgt_pad_masks = batch_tgt_pad_masks.to(device)
        
        backbone_out = backbone.forward(batch_inputs)
        backbone_out = backbone_out.to(device)
        sos_idx = word2idx['<sos>']
        sos_vector = torch.Tensor([sos_idx]*batch_targets.size(0)).view(-1,1).to(device)
        sos_batch_targets = torch.cat([sos_vector, batch_targets[:,:-1]], axis=1).type(torch.LongTensor).to(device)
        out = transformer.forward(backbone_out, sos_batch_targets, batch_in_pad_masks, batch_tgt_pad_masks)

        # Backpropagation
        loss = loss_fn(out.view(-1,vocab_size), batch_targets.view(-1))
        backbone_optimizer.zero_grad()
        transformer_optimizer.zero_grad()
        loss.backward()
        backbone_optimizer.step()
        transformer_optimizer.step()

        # Metrics:
        batch_acc = calc_batch_accuracy(out, batch_targets)
        all_accs.append(batch_acc)
        epoch_accs.append(batch_acc)

        tot_loss += loss.item()
        count += 1
        if count % cfg.VERBOSE_ITERS == 0:
            calc_batch_accuracy(out, batch_targets, verbose=True)
        t2 = datetime.datetime.now()
        print(f'Iteration time: {t2-t1} | loss: {loss.item():.3f} | acc: {round(batch_acc*100,2)}%')
        losses.append(tot_loss/count)
        all_loss.append(loss.item())

        iters_to_scheduler = 50
        # if len(all_accs) % iters_to_scheduler == 0:
        #     scheduler_acc = sum(all_accs[-iters_to_scheduler:])/iters_to_scheduler
        #     print(f'scheduler_acc: {scheduler_acc}')
        #     backbone_cheduler.step(scheduler_acc)
        #     transformer_cheduler.step(scheduler_acc)
        if len(all_loss) % iters_to_scheduler == 0:
            scheduler_loss = sum(all_loss[-iters_to_scheduler:])/iters_to_scheduler
            print(f'scheduler_loss: {scheduler_loss}')
            backbone_cheduler.step(scheduler_loss)
            transformer_cheduler.step(scheduler_loss)
        
    print(f'----- Epoch Metrics -----')
    print(f'Average epoch loss: {tot_loss/count:.3f}')
    print(f'Average epoch acc: {sum(epoch_accs)/len(epoch_accs)}')
    print(f'')
    # if (epoch+1) % cfg.INCREASE_DIFFICULTY_ITERS == 0 and difficulty_increases < cfg.MAX_INCREASES:
    #     print('\n!!!!!!!!!!!!!! Increasing difficulty !!!!!!!!!!!!!!')
    #     difficulty_increases += 1
    #     loader.increase_difficulty()


write_metric(losses, 'metrics/losses.txt')
write_metric(all_accs, 'metrics/accs.txt')


transformer_path = 'trained_models/transformer'
backbone_path = 'trained_models/backbone'
torch.save(transformer.state_dict(), transformer_path)
torch.save(backbone.state_dict(), backbone_path)