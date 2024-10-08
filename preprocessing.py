torch.cuda.empty_cache()

import gc
gc.collect()

#==================

class_arrays = [
    "Cassava Bacterial Blight (CBB)",
    "Cassava Brown Streak Disease (CBSD)",
    "Cassava Green Mottle (CGM)",
    "Healthy",
    "Cassava Mosaic Disease (CMD)"
]

#==================

train_transforms  = T.Compose([
  T.Resize((448, 448)),
  T.ToTensor(),
  T.RandomHorizontalFlip(0.5),
  T.RandomVerticalFlip(0.5),
  T.Normalize([0.4249, 0.4910, 0.3032], [0.2222, 0.2263, 0.2158])
])

valid_transforms  = T.Compose([
  T.Resize((448, 448)),
  T.ToTensor(),
  T.Normalize([0.4249, 0.4910, 0.3032], [0.2222, 0.2263, 0.2158])
])

#==================

class CustomDataSet(Dataset):
  def __init__(self, path, transform, data_type):
    self.path = path
    self.transform = transform
    
    self.l= 0
    self.x = []
    self.y = []
    
    class_arrays = [
        "Cassava___bacterial_blight",
        "Cassava___brown_streak_disease",
        "Cassava___green_mottle",
        "Cassava___healthy",
        "Cassava___mosaic_disease"
    ]
    
    cnt = 0
    
    train_len = int(1080*0.8)
    
    for i in class_arrays:
        P = glob.glob(path+'/'+i+'/*.jpg')
        
        if data_type == 'train':
            P = P[:train_len]
            
        else:
            P = P[train_len:1081]
        
        self.l += len(P)
        self.x += P
        self.y += [cnt]*len(P)
        
        cnt += 1
    
    print(self.l)
    
  def __len__(self):
    return self.l

  def __getitem__(self, idx):
    img_path = self.x[idx]
    label = self.y[idx]
    img = Image.open(img_path)
    img = img.convert('RGB')
    if self.transform is not None:
      img = self.transform(img)

    return img, label

#==================

p = Image.open('/kaggle/input/cassava-leaf-disease-classification/data/Cassava___bacterial_blight/1026467332.jpg')

plt.imshow(p)
plt.show()

#==================

path = '/kaggle/input/cassava-leaf-disease-classification/data'

#==================

train = []
valid = []

for i in range(1):
    train.append([])
    valid.append([])

train[0] = CustomDataSet(path = path, transform = train_transforms, data_type='train')
valid[0] = CustomDataSet(path = path, transform = valid_transforms, data_type='val')


print("train len : ", train[0].__len__())
print("valid len : ", valid[0].__len__())


img, label = train[0].__getitem__(10)
img = img.permute(1, 2, 0)
print(img.size())
plt.imshow(img)
plt.show()
print(label)
