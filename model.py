device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available')
#======================
# optimizer : 최적화 알고리즘
# criterion : 손실

def Train(net, epoch, optimizer, criterion, train_dataloader):
    print('[ Train epoch: %d ]' % epoch)
    net.train() # 모델을 학습 모드로 설정
    train_loss, correct = 0, 0
    total = 0
    cnt = 0
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        cnt += 1
        if cnt%10==0:
            print(int(cnt/10), ".    model is still training...")

    print('    Train accuarcy:', 100. * correct / total)
    print('    Train average loss:', train_loss / total)
    return (100. * correct / total, train_loss / total)


def Validate(net, epoch, criterion, val_dataloader):
    print('[ Validation epoch: %d ]' % epoch)
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    cnt = 0
    for batch_idx, (inputs, targets) in enumerate(val_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = net(inputs)
        val_loss += criterion(outputs, targets).item()
        _, predicted = outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        cnt += 1
        if cnt%100==0:
            print(int(cnt/100), ".    model is still evaluating...")

    print('    Accuarcy:', 100. * correct / total)
    print('    Average loss:', val_loss / total)
    return (100. * correct / total, val_loss / total)
#======================
from torchvision import models
import time

if not os.path.exists("saved"):
  os.makedirs("saved")
#사용하려는 모델에 따라 선택
#1. net = models.alexnet(pretrained=True).to(device)
#2. net = models.alexnet(pretrained=False).to(device)

#3. net = models.vgg16(pretrained=True).to(device)
#4. net = models.vgg16(pretrained=False).to(device)

#5. net = models.densenet121(pretrained=True).to(device)
#6. net = models.densenet121(pretrained=False).to(device)

#7. net = models.resnet101(pretrained=True).to(device)
#8. net = models.resnet101(pretrained=False).to(device)

#9. net = models.efficientnet_b5(pretrained=True).to(device)
#10. net = models.efficientnet_b5(pretrained=False).to(device)

#11. net = models.vit_b_16(pretrained=True).to(device)
#12. net = models.vit_b_16(pretrained=False).to(device)

#13. net = models.mobilenet_v2(pretrained=True).to(device)
#14. net = models.mobilenet_v2(pretrained=False).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4, betas = (0.9, 0.999), eps = 1e-8)


train_data = train[i]
valid_data = valid[i]
train_dataloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, drop_last=False)
#gpu memory 여부에 따라 batch_size 조정
valid_dataloader = DataLoader(dataset=valid_data, batch_size=1, shuffle=True, drop_last=False)


print('Training Dataset size:', len(train[i]))
print('Validation Dataset size:', len(valid[i]))

train_result = []
val_result = []

# 학습 시작
total_time = 0

for e in range(epoch):
    
  start_time=time.time()
  train_acc, train_loss = Train(net, e + 1, optimizer, criterion, train_dataloader) # 학습(training)
  val_acc, val_loss = Validate(net, e + 1, criterion, valid_dataloader) # 검증(validation)
  
  end_time=time.time()
  epoch_duration=end_time-start_time

  total_time += epoch_duration


  train_result.append((train_acc, train_loss))
  val_result.append((val_acc, val_loss))
  

  print(f"Epoch {e+1} - Duration: {epoch_duration:.2f} seconds")
  print(f"Total Duration: {total_time:.2f} seconds")
  print(f"Average Duration: {total_time/(e+1): .2f} seconds")
  print("train: ")
  print(train_result)
  print("val: ")
  print(val_result)

  torch.save(net, os.path.join('saved', "model.pt"))
#======================
