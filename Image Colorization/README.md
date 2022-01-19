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
* Vì mỗi mạng đều làm việc với mục tiêu khác nhau vì vậy cần thiết kế hàm loss cho mỗi mạng






* **Một số notebook tham khảo**
  * [LINK](https://colab.research.google.com/github/moein-shariatnia/Deep-Learning/blob/main/Image%20Colorization%20Tutorial/Image%20Colorization%20with%20U-Net%20and%20GAN%20Tutorial.ipynb) 
