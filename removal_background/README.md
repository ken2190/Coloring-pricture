# Một số kiến trúc mạng triển khai cho bài toán:
### [1. Kiến trúc mạng Unet](https://arxiv.org/pdf/1505.04597.pdf)
Kiến trúc mạng có 2 phần cơ bản: encoder, decoder

Trong đó phần encoder dùng để giảm chiều dài và chiều rộng của ảnh bằng việc sử dụng các lớp convolutions và các lớp poolings. Trong đó phần decoder dùng để phục hồi lại kích thước ban đầu của ảnh. 

Phần encoder thường chỉ là một mạng CNN thông thường nhưng bỏ đi những layer fully conected cuối cùng. Chúng ta có thể sử dụng những mạng có sẵn trong phần encoder như VGG16, VGG19, Alexnet,... Còn decoder tùy vào các kiến trúc mạng mà ta có thể xây dựng khác nhau.

![alt](https://i.imgur.com/lKZGO0C.png)

### 2. Kiến trúc mạng VGG16-UNET
[Notebook(ver8-9-10)](https://www.kaggle.com/acousticmusic/unet-removal-background-ver1)

Phần encoder sử dụng mạng CNN VGG16 dùng extract features.

Phần decoder sử dụng Unet


### [3. Kiến trúc mạng U2NET-U2NETP](https://arxiv.org/abs/2005.09007)
#### 3.1 Residual U-blocks
Thông tin ngữ cảnh toàn cục (global) và cục bộ (local) rất quan trọng trong các bài toán về object detection and other segmentation tasks. Trong các modern CNN designs, such as VGG,Desnest,ResNet,so on. Với bộ các filters convolutional có kích thước 1x1, 3x3 thường xuyên được sử dụng trong việc trích xuất đặc trưng. Như vậy thì sẽ hiệu quả trong việc lưu trữ và mặt tính toán. Output sẽ chứa các local features bởi vì trường tiếp nhận của các bộ lọc 1x1, 3x3 quá nhỏ để để nắm thông tin toàn cục (global). Để thu thập được nhiều thông tin toàn cục (global) ở các features map có độ phân giải cao hơn từ các lớp nông. Ỷ tưởng tốt nhấy là phóng to trường tiếp nhận. 

![image](https://user-images.githubusercontent.com/72034584/153568348-b0d4d9a6-5814-40d0-8244-f71ebf6f09ce.png) 

Ở hình phía trên chúng ta có thể extract both local and non-local features bằng cách mở rộng trường tiếp nhận sử dụng cách là giãn nỡ convolution(dilated convolutional). Tuy nhiên, việc tiến hành giãn nở convolution nhiều trên input feature map(nhất là giải đoạn đầu) với độ phân giải gốc phải yêu cầu chi phí lớn về mặt tính toán. 

![image](https://user-images.githubusercontent.com/72034584/153591013-66ed94de-193e-4fa8-9064-876029e847cd.png)

Cảm hứng từ vởi Unet, có phương pháp được đề xuất Residual U-blocks, để nắm bắt đặc trưng đa quy mô trong giai đoạn. Nhìn hình phía trên ta có thể thấy, RSU-L(C_in,M,C_out), L ở đây là số layers trong phần encoder. C_in,C_out là biểu thị input,output channels. M biểu thị số channels trong các lớp bên trong RSU. 

#### 3.2 Kiến trúc U2NET

![image](https://user-images.githubusercontent.com/72034584/153592006-06b156d9-7284-4772-bdfd-636ad62d0e5c.png)

Chúng ta có thể thấy kiến trúc hình dạng chữ U. Và có phần như đã đề cập ở trên encoder, decoder.

**Trong phần encoder**: En_1,En_2,En_3,En_4 chúng ta sử dụng Residual U-blocks RSU-7,RSU-6,RSU-5,RSU-4, như nói trước đó 7,6,5,4 biểu thị chiều cao của RSU (Layers). L thường được cấu hình theo độ phân giải không gian của các input feature map.  Đối với feature maps có chiều rộng,cao lớn thì chúng ta sử dụng L lớn để thu thập thông tin ngữ cảnh hơn. En_5,En_6 có độ phân giải feature map khá thấp, Vì vậy ở 2 giai đoạn này chúng ta sử dụng RSU-4F (F có nghĩa là phiên bản giãn nở), chúng ta thay thể việc pooling and upsampling bằng dilated convolution. 

**Trong phần decoder**: Cũng tương tự như bên encoder, De_5,De_4,De_3,De_2,De_1. Trong De_5, chúng ta sử dụng dilated version residual U-block RSU-4F tương tự như bên En_5,En_6. 
Ở mỗi giai đoạn decoder lấy concatenation umsampled feature maps from its previous stage. Và các feature map của giai đoạn encoder đối xứng làm input đầu vào. 

**Phần output**: U2NET tạo ra đầu ra 6 bên: S_6,S_5,S_4,S_3,S_2,S_1 tương đương En_6,De_5,De_4,De_3,De_2,De_1 bởi 3x3 convolution và 1 sigmoid function. 

Tóm lại, thiết kế của U2 -Net cho phép có kiến trúc sâu với các tính năng đa quy mô phong phú và chi phí tính toán và bộ nhớ tương đối thấp. 


### 4. Một số notebook và một số kết quả thu được từ mạng VGG16-Unet

#### Dataset: cocopersonsegmentation hoặc person-segmentation-dataset

#### Một số kết quả khá quan:

|Input                                  |Output                                       |
|-------------------------------------- |---------------------------------------------|
|![ros](https://user-images.githubusercontent.com/72034584/153563915-23280981-3e30-4c29-93ea-a0b6dc9d5fdb.jpg)|![result_img](https://user-images.githubusercontent.com/72034584/153563950-37dba6e7-fb81-428a-8746-d632c194fe1d.jpg)|
|![roses5](https://user-images.githubusercontent.com/72034584/153564138-f99d0449-66f3-4137-a12a-eeabb27f18b6.jpg)|![result_img1](https://user-images.githubusercontent.com/72034584/153564153-eb68445a-fdc5-4c6e-a61c-cb2abe8aff63.jpg)|
|![qq1](https://user-images.githubusercontent.com/72034584/153564024-1d76f480-36bf-40a8-9513-2ae8b9838250.jpg)|![result_img2](https://user-images.githubusercontent.com/72034584/153564038-ce3a439a-7863-4097-97b3-ce87f2d4cf5d.jpg)|



### 5. Một số notebook và một số kết quả thu được từ mạng U2NET

|Input                                  |Output                                       |
|-------------------------------------- |---------------------------------------------|
|![ros](https://user-images.githubusercontent.com/72034584/153563915-23280981-3e30-4c29-93ea-a0b6dc9d5fdb.jpg)|![ross](https://user-images.githubusercontent.com/72034584/153599158-bde94f5b-6555-48a7-9612-1d5ac369e9ec.jpg)|
|![roses5](https://user-images.githubusercontent.com/72034584/153564138-f99d0449-66f3-4137-a12a-eeabb27f18b6.jpg)|![tải xuống](https://user-images.githubusercontent.com/72034584/153599439-9eb52877-dd2d-4734-b8cf-cb2a0823229e.jpg)|
|![photo-1-15902508724041176556397](https://user-images.githubusercontent.com/72034584/153599841-ce642fa0-14c3-4d2e-ab4e-9d058c09141c.jpg)|![tải xuống (1)](https://user-images.githubusercontent.com/72034584/153600030-1792eb2b-4355-4047-b534-f4baeae8564f.jpg)|



#### Posts
* [Bài viết:Background removal with deep learning](https://towardsdatascience.com/background-removal-with-deep-learning-c4f2104b3157)

