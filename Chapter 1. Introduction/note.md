# 1. Quy trình làm việc liên quan đến machine learning
* Data collecting ![formula](https://render.githubusercontent.com/render/math?math=\rightarrow) data preprocessing (2) $\rightarrow$ data analysis, build model $\rightarrow$ test, kiểm tra xem model nào là phù hợp _(có lặp lại khi model dự đoán ko chính xác)_
* Có một mảng nghiên cứu riêng danh cho việc tạo ra dữ liệu
* Supervised Learning: là các bài toán cần dự đoán một thứ gì đó mà có target.
  * Dc chia làm 2 bài toán chính là: 
    * Classification _(Phân lớp)_, ví dụ một email là thì dự đoán nó là email spam hay thường.
    * Regression _(Dự báo, thường là dữ liệu số nào đó như giá nhà)_.
* Unsupervised Learning: là các bài toán ko cần đưa ra target
* Reinforcement Learning: là máy tự học, thử và sai lâu dần rút ra tri thức, ví dụ áp dụng máy để chơi các game chơi cờ, flappy bird, tức là máy tự học dc sau quá trình làm việc