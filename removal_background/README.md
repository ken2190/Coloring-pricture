# Một số kiến trúc mạng triển khai:
## Bài toán semantics segmentation
### [1. Kiến trúc mạng Unet](https://arxiv.org/pdf/1505.04597.pdf)
Kiến trúc mạng có 2 phần cơ bản: encoder, decoder

Trong đó phần encoder dùng để giảm chiều dài và chiều rộng của ảnh bằng việc sử dụng các lớp convolutions và các lớp poolings. Trong đó phần decoder dùng để phục hồi lại kích thước ban đầu của ảnh. 

Phần encoder thường chỉ là một mạng CNN thông thường nhưng bỏ đi những layer fully conected cuối cùng. Chúng ta có thể sử dụng những mạng có sẵn trong phần encoder như VGG16, VGG19, Alexnet,... Còn decoder tùy vào các kiến trúc mạng mà ta có thể xây dựng khác nhau.

![alt](https://i.imgur.com/lKZGO0C.png)


### [2. Kiến trúc mạng Tiramisu](https://arxiv.org/pdf/1611.09326.pdf)

* Kiến trúc mạng

![alt](https://miro.medium.com/max/694/0*8y3DsK9cGoW9tpne.) -------> ![Screenshot 2022-01-20 223450](https://user-images.githubusercontent.com/72034584/150370352-499f67c9-1d87-48c5-857f-cc8ddd4f6a6c.png)

![Screenshot 2022-01-20 223450](https://user-images.githubusercontent.com/72034584/150372015-5adb61f7-bdcd-4bc9-8774-ca478626ced8.png)

Mô hình tiramisu là một mô hình từ một khóa học của Jeremy Howard’s mới ra đời. Tên đầy đủ nó là "100 layers Tiramisu" ngụ ý là nó rất lớn, nhưng thực tế nó không phức tạp như vậy vì nó 9-10 triệu tham số. Mô hình tiramisu dựa trên mô hình Densnet. Và nó thêm phân skip-connections ở lớp up-sampling giống như mạng Unet.  

Mô hình Densnet là được phát triển từ mô hình resnet, nhưng thay thế "ghi nhớ" cho lớp tiếp theo. Densnet lại có thể ghi nhớ tất cả các lớp xuyên suốt mô hình. Các kết nối này được gọi là kết nối đường cao tốc. Nó gây ra lạm phát số bộ lọc, được định nghĩa là "tốc độ tăng trưởng". Tiramisu có tốc độ phát triển là 16, do đó, với mỗi lớp, chúng tôi thêm 16 bộ lọc mới cho đến khi chúng tôi đạt đến các lớp 1072 bộ lọc. Bạn có thể mong đợi 1600 lớp vì đó là 100 lớp tiramisu, tuy nhiên, các lớp lấy mẫu lên sẽ làm giảm một số bộ lọc.


Chúng ta sẽ train mô hình giống như mô tả trong [paper](https://arxiv.org/pdf/1611.09326.pdf) standard cross entropy loss, RMSProp optimizer with 1e-3 learning rate and small decay. Khởi tạo trọng số HeUniform, 

```python
#Layer
def conv_block(x,n_filters,filter_size = 3, drop_out = 0.2):
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(n_filters,filter_size,padding='same',kernel_initializer='he_uniform')(x)
  if drop_out != 0:
    x = Dropout(drop_out)(x)
  return x
```

```python
#dense_block bên encoder
def down_dense_block(x,growth_rate, blocks, drop_out):
  for i in range(blocks):
    x1 = conv_block(i,growth_rate,filter_size=3,drop_out=drop_out)
    x = Concatenate(axis = 3)([x,x1])
  return x
  
#dense_block bên decoder
def up_dense_block(x,growth_rate, blocks, drop_out):
  block_to_upsample = []
  for i in range(blocks):
    x1 = conv_block(i,growth_rate,filter_size=3,drop_out=drop_out)
    block_to_upsample.append(x1)
    x = Concatenate(axis=3)([x, x1])
  return Concatenate(axis=3)(block_to_upsample)
```

```python
def transition_down(x, n_filters, dropout):
    x = conv_block(x, n_filters, filter_size=1, dropout=dropout)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    return x
```

```python
def transition_up(x, skip_connection, n_filters_keep):
    x = Conv2DTranspose(filters=n_filters_keep, kernel_size=3, strides=(2, 2), padding='same', activation='relu',
                        kernel_initializer='he_uniform')(x)
    x = Concatenate(axis=3)([x, skip_connection])
    return x
```

* [Notebook](https://files.fast.ai/part2/lesson14/)
* Một số kết quả khá quan hơn Unet
![image](https://user-images.githubusercontent.com/72034584/150340219-2df1e1eb-9589-4e2f-a090-fd1a460a458c.png)

















### 3. Một số notebook và một số kết quả thu được từ mạng Unet
#### [Đây là link kaggle để chạy thử notebook này](https://www.kaggle.com/acousticmusic/unet-removal-background-ver1)

#### Dataset: cocopersonsegmentation hoặc person-segmentation-dataset

#### Một số kết quả khá quan:
Vì mới trên 1 epoch nên kết quả chưa được tốt, có thể tăng số epoch lên chắc sẽ khá quan hơn.

Cột ảnh đầu tiên là ảnh gốc ban đầu, cột thứ 2 là ảnh dự đoán và chèn background mới, cột thứ 3 là dự đoán

![image](https://user-images.githubusercontent.com/72034584/149666138-9d4c3a96-3b2d-4e82-8137-13513fb8b5fc.png)

![image](https://user-images.githubusercontent.com/72034584/149666148-52af094f-6531-4238-b657-0719730ebca3.png)

![image](https://user-images.githubusercontent.com/72034584/149666157-16c04a82-5012-4aae-b539-e8c3f838e7b6.png)

![image](https://user-images.githubusercontent.com/72034584/149666178-2b960a1a-db01-44af-82b6-e5798ff293a9.png)

![image](https://user-images.githubusercontent.com/72034584/150349372-3a9f7a71-0cc4-461f-ac32-c8f52ec1e96f.png)

![image](https://user-images.githubusercontent.com/72034584/150349302-3ffc4b1e-e9ef-4267-9aac-df5c6e89ae8e.png)

![image](https://user-images.githubusercontent.com/72034584/150354486-6489cafe-8a41-4a6f-8f83-caac9caa5cb9.png)

![image](https://user-images.githubusercontent.com/72034584/150356631-cbdda2f4-0c99-4b00-b7e2-d2056b1120af.png)


#### Posts
* [Bài viết:Background removal with deep learning](https://towardsdatascience.com/background-removal-with-deep-learning-c4f2104b3157)

