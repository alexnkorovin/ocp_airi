def send_scalars(lr, loss, writer, step=-1, epoch=-1, type_='train'):
    if type_ == 'train':
        writer.add_scalar('lr per step on train', lr, step) 
        writer.add_scalar('loss per step on train', loss, step)
    if type_ == 'val':
        writer.add_scalar('loss per epoch on val', loss, epoch)

def send_hist(model, writer, step):
    for name, weight in model.named_parameters():
        writer.add_histogram(name, weight, step)

def train(model, iterator, optimizer, criterion, print_every=10, epoch=0, writer=None, device='cpu'):
    
    print(f'epoch {epoch}')
    
    epoch_loss = 0
    
    model.train()

    for i, (systems, ys) in enumerate(iterator):
        
        optimizer.zero_grad()
        predictions = model(systems).squeeze()
        
        loss = criterion(predictions.float(), ys.to(device).float())
        loss.backward()     
        
        optimizer.step()
        
        batch_loss = loss.item() 
        epoch_loss += batch_loss  
        
        if writer != None:
            
            lr = optimizer.param_groups[0]['lr']
            
            step = i + epoch*len(iterator)
            
            send_hist(model, writer, i)
            send_scalars(lr, batch_loss, writer, step=step, epoch=epoch, type_='train')
        
        if not (i+1) % print_every:
            print(f'step {i} from {len(iterator)} at epoch {epoch}')
            print(f'Loss: {batch_loss}')
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, epoch=0, writer=False, device='cpu'):
    
    print(f'epoch {i} evaluation')
    
    epoch_loss = 0
    
#    model.train(False)
    model.eval()  
    
    with torch.no_grad():
        for systems, ys in iterator:   

            predictions = model(systems).squeeze()
            loss = criterion(predictions.float(), ys.to(device).float())        

            epoch_loss += loss.item()
            
    overall_loss = epoch_loss / len(iterator)

    if writer != None:
        send_scalars(None, overall_loss, writer, step=None, epoch=epoch, type_='val')
                
    print(f'epoch loss {overall_loss}')
    print('========================================================================================================')
            
    return overall_loss

def inference(model, iterator):
    y = torch.tensor([])

#    model.train(False)
    model.eval()  
    
    with torch.no_grad():
        for systems, ys in iterator:   
          predictions = model(systemhs).squeeze()
          y = torch.cat((y, predictions))
      
    return y