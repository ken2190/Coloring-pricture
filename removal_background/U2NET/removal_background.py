from model import *
from load_data import *
import matplotlib.pyplot as plt
from PIL import Image


#Load weight model
model_dir = '../U2NET/weights/weights_u2netp/u2netp_1.pth'

net = U2NETP(3, 1)
net.load_state_dict(torch.load(model_dir))
if torch.cuda.is_available():
    net.cuda()
net.eval()


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn

def preprocess(image):
    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[0:2])

    if (3 == len(label_3.shape)):
        label = label_3[:, :, 0]
    elif (2 == len(label_3.shape)):
        label = label_3

    if (3 == len(image.shape) and 2 == len(label.shape)):
        label = label[:, :, np.newaxis]
    elif (2 == len(image.shape) and 2 == len(label.shape)):
        image = image[:, :, np.newaxis]
        label = label[:, :, np.newaxis]

    transform = transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
    sample = transform({
        'imidx': np.array([0]),
        'image': image,
        'label': label
    })

    return sample


def run(img):
    torch.cuda.empty_cache()

    sample = preprocess(img)
    inputs_test = sample['image'].unsqueeze(0)
    inputs_test = inputs_test.type(torch.FloatTensor)

    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
    else:
        inputs_test = Variable(inputs_test)

    d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

    # Normalization.
    pred = d1[:, 0, :, :]
    predict = normPRED(pred)

    # Convert to PIL Image
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np * 255).convert('RGB')
    
    print(predict_np.shape)
    
    # Cleanup.
    del d1, d2, d3, d4, d5, d6, d7

    return im,predict_np

img = Image.open('../img_test/ros.jpg')
img1,size = run(np.array(img))
mask = img1.convert('L').resize((img.width, img.height))
empty = Image.new("RGBA", img.size, 0)
img = Image.composite(img, empty, mask)
plt.figure(figsize = (40,40))
plt.imshow(img)