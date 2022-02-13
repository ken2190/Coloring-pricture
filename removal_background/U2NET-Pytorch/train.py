from model import *
from load_data import *
from torchvision import transforms, utils
import torch
from PIL import Image
import torch.optim as optim
import os



#Các đường dẫn tới folder data
model_name = 'u2net' 
data_dir = os.path.join('data' + os.sep)
tra_image_dir = os.path.join('DUTS-TR', 'DUTS-TR-Image' + os.sep)
tra_label_dir = os.path.join('DUTS-TR', 'DUTS-TR-Mask' + os.sep)

image_ext = '.jpg'
label_ext = '.png'

model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)


#Danh sách path image
tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

#Danh sách path mask
tra_lbl_name_list = []
for img_path in tra_img_name_list:
	img_name = img_path.split(os.sep)[-1]

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)


#Load Data
salobj_dataset = SalObjDataset(img_name_list=tra_img_name_list,lbl_name_list=tra_lbl_name_list,
                    transform=transforms.Compose([
                            RescaleT(320),
                            RandomCrop(288),
                            ToTensorLab(flag=0)]))
batch_size_train = 12
salobj_dataloader = DataLoader(salobj_dataset, batch_size = batch_size_train, shuffle=True, num_workers=1)



#Khởi tạo model
net = U2NETP(3, 1)
if torch.cuda.is_available():
    net.cuda()

optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

#Các tham số khởi tạo
epoch_num = 20
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
train_num = len(tra_img_name_list)
val_num = 0 
save_frq = 1000 # Save model every 1000 iterations


for epoch in range(0, epoch_num):
    net.train()
    for i, data in enumerate(salobj_dataloader):

        inputs, labels = data['image'], data['label']
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        optimizer.step()

        running_loss += loss.data.item()
        running_tar_loss += loss2.data.item()

        del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
        if ite_num % save_frq == 0:

            torch.save(net.state_dict(), "model_u2netp_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train() 
            ite_num4val = 0



