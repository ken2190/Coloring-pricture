## Tìm hiểu về màu sắc.

* Màu RGB:

    ![alt](https://raw.githubusercontent.com/moein-shariatnia/Deep-Learning/main/Image%20Colorization%20Tutorial/files/rgb.jpg)

Như chúng ta đã biết, khi chúng ta load một bức ảnh màu thì chúng ta sẽ dễ dàng nhìn thấy có 3 kênh màu: Đỏ, Xanh lá cây, Xanh dương. Hay có thể gọi màu RGB, chúng ta sẽ biết mỗi pixel có màu gì tương ứng. 

* Màu Lab:

![alt](https://raw.githubusercontent.com/moein-shariatnia/Deep-Learning/main/Image%20Colorization%20Tutorial/files/lab.jpg)

Trong không gian màu L * a * b, chúng ta lại có ba số cho mỗi pixel nhưng những con số này có ý nghĩa khác nhau. L đại diện cho độ sáng của mỗi pixel, Các kênh a và b mã hóa mỗi pixel có giá trị tương ứng là bao nhiêu màu xanh lục-đỏ và vàng-xanh lam.

**Chúng ta thường sẽ sử dụng màu Lab để traning mô hình**. Để đào tạo một mô hình để tô màu, chúng ta nên cung cấp cho nó một hình ảnh thang độ xám và hy vọng rằng nó sẽ làm cho nó có màu sắc. Nếu chúng ta sử dụng màu Lab, thì nó cung cấp cho kênh L(độ sáng, là hình ảnh thang độ xám) và chúng ta chỉ cần đưa dự đoán với 2 kênh còn lại đó là a và b. Sau khi dự đoán thì chúng ta ghép các kênh lại và thu được một bức ảnh hoàn chỉnh. Còn nếu chúng ta sử dụng màu RGB, thì chúng ta cần chuyển ảnh màu về ảnh xám, và chúng ta cần đưa ra dự đoán với 3 kênh r,g,b như vậy sẽ khó hơn. 

## Tìm hiểu mạng GAN

* Như chúng ta đã được học thì một mạng GAN có phần cơ bản: generator và discriminator(sinh sản và phân biệt), tùy nhiên cần nhớ rằng là 2 mạng này cần hoạt động cùng nhau. Thật ra ỷ tưởng GAN bắt nguồn từ zero-sum non-cooperative game, hiểu đơn giản như trò chơi đối kháng 2 người (cờ vua, cờ tướng). Ở mỗi lượt thì cả 2 đều muốn maximize cơ hội thắng của tôi và minimize cơ hội thắng của đối phương. Discriminator và Generator trong mạng GAN giống như 2 đối thủ trong trò chơi. 
* Vì mỗi mạng đều làm việc với mục tiêu khác nhau vì vậy cần thiết kế hàm loss cho mỗi mạng. Tuy nhiên trong bài toán chúng ta đang làm đây thì sẽ L1-LOSS cho các 2 mạng.
* Chúng ta sẽ triển khai giống bài toán pix2pix(image-to-image-Translation with Conditional Adversarial Networks)
* |[Paper](https://arxiv.org/abs/1611.07004)|[Demo pix2pix](https://affinelayer.com/pixsrv/)|
* Triển khai mạng GAN trong bài toán này:
  * Mạng Generator sẽ lấy hình ảnh xám thang đo L, và tạo ra 2 kênh a,b
  * Mạng Discriminator sẽ lấy 2 kênh được tạo ra đó ghép với hình ảnh xám đầu(Kênh L) và quyết định 3 kênh mới này giả hay thật
  * Tất nhiên Mạng Discriminator cũng cần xem một số hình ảnh thực (hình ảnh 3 kênh một lần nữa trong không gian màu Lab) không phải được tạo ra và nên mới biết rằng chúng là thật. 
  * Hình ảnh thang độ xám mà cả Generator và Discriminator đều nhìn thấy là điều kiện mà chúng tôi cung cấp cho cả hai mô hình trong GAN của chúng tôi và mong rằng chúng sẽ xem xét điều kiện này.
  * Hàm Loss Gan:
    ![alt](https://raw.githubusercontent.com/moein-shariatnia/Deep-Learning/main/Image%20Colorization%20Tutorial/files/GAN_loss.jpg) 
    
    * x là ảnh xám, z là input noise cho mạng Generator, y là output 2 kênh màu a,b, G là mạng Generator, D là discriminator
  * Chúng ta sẽ kết hợp chức năng mất mát với Mất mát L1 (bạn có thể biết tổn thất L1 là sai số tuyệt đối trung bình) của các màu dự đoán so với màu thực tế:
    ![alt](https://raw.githubusercontent.com/moein-shariatnia/Deep-Learning/main/Image%20Colorization%20Tutorial/files/l1_loss.jpg)
  * Nếu chúng ta chỉ sử dụng độ mất L1, mô hình vẫn học cách chỉnh màu hình ảnh nhưng nó sẽ mang tính bảo thủ và phần lớn thời gian sử dụng các màu như "xám" hoặc "nâu" vì khi nó nghi ngờ màu nào là tốt nhất, nó sẽ lấy giá trị trung bình và sử dụng những màu này để giảm tổn thất L1 nhiều nhất có thể (nó tương tự như hiệu ứng làm mờ của mất L1 hoặc L2 trong tác vụ siêu phân giải). Ngoài ra, Suy hao L1 được ưa thích hơn Suy hao L2 (hoặc sai số bình phương trung bình) vì nó làm giảm tác động của việc tạo ra hình ảnh màu xám. Vì vậy, hàm mất mát kết hợp của chúng tôi sẽ là:
    ![alt](https://raw.githubusercontent.com/moein-shariatnia/Deep-Learning/main/Image%20Colorization%20Tutorial/files/loss.jpg)
* **Một số notebook tham khảo**
  * [LINK 1](https://colab.research.google.com/drive/1EUdLJpo39zMW6JLhya4_E1nJQ4WegZVl?authuser=1#scrollTo=Q_Unpb58dohK)
  * [LINK 2](https://colab.research.google.com/github/moein-shariatnia/Deep-Learning/blob/main/Image%20Colorization%20Tutorial/Image%20Colorization%20with%20U-Net%20and%20GAN%20Tutorial.ipynb) 
* **Một số video tham khảo**
  * [VIDEO](https://www.youtube.com/watch?v=v88IUAsgfz0) 

* **Một số bài viết tham khảo**
  * [POSTS 1](https://blog.floydhub.com/colorizing-and-restoring-old-images-with-deep-learning/)
  * [POSTS 2](https://blog.floydhub.com/colorizing-b-w-photos-with-neural-networks/) 

