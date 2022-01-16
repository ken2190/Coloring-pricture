### Bài toán này, có thể sử dụng dạng bài Semantics Segmentation. Và trong bài t sử dụng kiến trúc Unet, để phân vùng ảnh trên mỗi pixel. Về cơ bản nó sẽ hoạt động như vậy.

Kiến trúc mạng có 2 phần cơ bản: encoder, decoder

Trong đó phần encoder dùng để giảm chiều dài và chiều rộng của ảnh bằng việc sử dụng các lớp convolutions và các lớp poolings. Trong đó phần decoder dùng để phục hồi lại kích thước ban đầu của ảnh. 

Phần encoder thường chỉ là một mạng CNN thông thường nhưng bỏ đi những layer fully conected cuối cùng. Chúng ta có thể sử dụng những mạng có sẵn trong phần encoder như VGG16, VGG19, Alexnet,... Còn decoder tùy vào các kiến trúc mạng mà ta có thể xây dựng khác nhau.

![alt](https://i.imgur.com/lKZGO0C.png)

#### [Đây là link kaggle để chạy thử notebook này](https://www.kaggle.com/acousticmusic/unet-removal-background-ver1)

#### Dataset: cocopersonsegmentation hoặc person-segmentation-dataset

#### Một số kết quả khá quan:
Vì mới trên 1 epoch nên kết quả chưa được tốt, có thể tăng số epoch lên chắc sẽ khá quan hơn

![image](https://user-images.githubusercontent.com/72034584/149666138-9d4c3a96-3b2d-4e82-8137-13513fb8b5fc.png)

![image](https://user-images.githubusercontent.com/72034584/149666148-52af094f-6531-4238-b657-0719730ebca3.png)

![image](https://user-images.githubusercontent.com/72034584/149666157-16c04a82-5012-4aae-b539-e8c3f838e7b6.png)

![image](https://user-images.githubusercontent.com/72034584/149666178-2b960a1a-db01-44af-82b6-e5798ff293a9.png)
