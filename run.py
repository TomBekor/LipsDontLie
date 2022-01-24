import torch
from torch import dtype, nn, optim
import datetime
from dataloader import LandmarksDataLoader, Tokenizer
from model import Backbone, LandmarksNN, Transformer
from utils import calc_batch_accuracy, write_metric
import config as cfg
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the vocabulary
vocab_file = open(cfg.VOCABULARY_FILE, 'r')
idx2word = vocab_file.read().splitlines()
vocab_size = len(idx2word)
word2idx = {word:idx for idx, word in enumerate(idx2word)}
vocab_file.close()

# Intialize DataLoaders and Tokenizer
train_loader = LandmarksDataLoader(landmarks_path=cfg.LANDMARKS_DIR, annotations_path=cfg.ANNOTATIONS_DIR,
 mode='train', batch_size=cfg.BATCH_SIZE, shuffle=True)

val_loader = LandmarksDataLoader(landmarks_path=cfg.LANDMARKS_DIR, annotations_path=cfg.ANNOTATIONS_DIR,
 mode='validation', batch_size=1, shuffle=True)

test_loader = LandmarksDataLoader(landmarks_path=cfg.LANDMARKS_DIR, annotations_path=cfg.ANNOTATIONS_DIR,
 mode='test', batch_size=cfg.BATCH_SIZE, shuffle=True)

tokenizer = Tokenizer(word2idx)

# Initialize the backbone and the transformer models
# backbone = Backbone()
# backbone = backbone.to(device)
landmarks_model = LandmarksNN()
landmarks_model.apply(landmarks_model.initialize_weights)
landmarks_model.to(device)
transformer = Transformer(vocab_size)
transformer = transformer.to(device)

# Initialize optimizers and loss function
# backbone_optimizer = optim.Adam(backbone.parameters())
landmarks_optimizer = optim.Adam(landmarks_model.parameters())
transformer_optimizer = optim.SGD(transformer.parameters(), lr=1e-1)
loss_fn=nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])

# backbone_scheduler = ReduceLROnPlateau(backbone_optimizer, mode='min', factor=0.1, patience=2, verbose=True)
landmarks_scheduler = ReduceLROnPlateau(landmarks_optimizer, mode='min', factor=0.1, patience=2, verbose=True)
transformer_scheduler = ReduceLROnPlateau(transformer_optimizer, mode='min', factor=0.1, patience=2, verbose=True)

losses = []
all_accs = []
all_loss = []
val_accs = []

# Training loop
# backbone.train()
#difficulty_increases = 0
for epoch in range(cfg.EPOCHS):
    print(f'<--------------------- Epoch: {epoch + 1} --------------------->')
    landmarks_model.train()
    transformer.train()
    epoch_accs = []
    tot_loss = 0
    count = 0
    for samples, labels in train_loader:
        t1 = datetime.datetime.now()
        tokenizer_res = tokenizer.tokenize(samples, labels)
        batch_inputs, batch_targets,\
        batch_in_pad_masks, batch_tgt_pad_masks = tokenizer_res

        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        batch_in_pad_masks = batch_in_pad_masks.to(device)
        batch_tgt_pad_masks = batch_tgt_pad_masks.to(device)
        
        # backbone_out = backbone.forward(batch_inputs)
        # backbone_out = backbone_out.to(device)
        landmarks_out = landmarks_model.forward(batch_inputs)
        landmarks_out = landmarks_out.to(device)
        sos_idx = word2idx['<sos>']
        sos_vector = torch.Tensor([sos_idx]*batch_targets.size(0)).view(-1,1).to(device)
        sos_batch_targets = torch.cat([sos_vector, batch_targets[:,:-1]], axis=1).type(torch.LongTensor).to(device)
        out = transformer.forward(landmarks_out, sos_batch_targets, batch_in_pad_masks, batch_tgt_pad_masks)

        # Backpropagation
        loss = loss_fn(out.view(-1,vocab_size), batch_targets.view(-1))
        landmarks_optimizer.zero_grad()
        transformer_optimizer.zero_grad()
        loss.backward()
        landmarks_optimizer.step()
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
        iter_time = str(t2-t1).split(':')[-1][:-3]
        print(f'Iteration time: {iter_time} | loss: {loss.item():.3f} | acc: {round(batch_acc*100,2)}%')
        losses.append(tot_loss/count)
        all_loss.append(loss.item())

        iters_to_scheduler = 50
        # if len(all_accs) % iters_to_scheduler == 0:
        #     scheduler_acc = sum(all_accs[-iters_to_scheduler:])/iters_to_scheduler
        #     print(f'scheduler_acc: {scheduler_acc}')
        #     backbone_scheduler.step(scheduler_acc)
        #     transformer_scheduler.step(scheduler_acc)
        if len(all_loss) % iters_to_scheduler == 0:
            scheduler_loss = sum(all_loss[-iters_to_scheduler:])/iters_to_scheduler
            print(f'scheduler_loss: {scheduler_loss}')
            landmarks_scheduler.step(scheduler_loss)
            transformer_scheduler.step(scheduler_loss)
        
    print(f'----- Epoch Metrics -----')
    print(f'Average epoch loss: {tot_loss/count:.3f}')
    print(f'Average epoch acc: {sum(epoch_accs)/len(epoch_accs)}')
    print(f'')
    # if (epoch+1) % cfg.INCREASE_DIFFICULTY_ITERS == 0 and difficulty_increases < cfg.MAX_INCREASES:
    #     print('\n!!!!!!!!!!!!!! Increasing difficulty !!!!!!!!!!!!!!')
    #     difficulty_increases += 1
    #     loader.increase_difficulty()

    # Validation Metrics:
    landmarks_model.eval()
    transformer.eval()
    with torch.no_grad():
        epoch_val_accs = []
        for samples, labels in val_loader:
            # Forward
            tokenizer_res = tokenizer.tokenize(samples, labels)
            batch_inputs, batch_targets,\
            batch_in_pad_masks, batch_tgt_pad_masks = tokenizer_res

            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            batch_in_pad_masks = batch_in_pad_masks.to(device)
            batch_tgt_pad_masks = batch_tgt_pad_masks.to(device)
            
            landmarks_out = landmarks_model.forward(batch_inputs)
            landmarks_out = landmarks_out.to(device)
            sos_idx = word2idx['<sos>']
            sos_vector = torch.Tensor([sos_idx]*batch_targets.size(0)).view(-1,1).to(device)
            sos_batch_targets = torch.cat([sos_vector, batch_targets[:,:-1]], axis=1).type(torch.LongTensor).to(device)
            out = transformer.forward(landmarks_out, sos_batch_targets, batch_in_pad_masks, batch_tgt_pad_masks)

            # Accuracy:
            batch_acc = calc_batch_accuracy(out, batch_targets)
            epoch_val_accs.append(batch_acc)
        total_batch_val_acc = sum(epoch_val_accs)/len(epoch_val_accs)
        val_accs.append(total_batch_val_acc)


write_metric(losses, 'metrics/tain_losses.txt')
write_metric(all_accs, 'metrics/tain_accs.txt')
write_metric(val_accs, 'metrics/val_accs.txt')

transformer_path = 'trained_models/transformer'
landmarks_path = 'trained_models/landmarks'
torch.save(transformer.state_dict(), transformer_path)
torch.save(landmarks_model.state_dict(), landmarks_path)