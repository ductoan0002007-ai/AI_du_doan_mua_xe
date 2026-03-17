import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
np.random.seed(42)
#B1:tạo dữ liệu 
age=np.random.randint(22,60,100)
salary=np.random.randint(15,120,100)

potential=age*1.5+salary*3.0
#normal(loc,scale,size) tạo 1 mảng các con số ngẫu nhiên dao động quanh 0 để cộng têm vào điểm gốc 
interuption=np.random.normal(0,50,100)
real=potential+interuption
#giống if else nếu >180 trả về 1,còn lại trả về 0  
buy=np.where(real>180,1,0)
data={
    "age":age,
    "salary":salary,
    "buy":buy
}

#chuyển dữ liệu thành bảng
df=pd.DataFrame(data)
#cố tình chèn thêm vài dữ liệu bị thiếu(NaN)
df.loc[3,"age"]=np.nan
df.loc[7,"salary"]=np.nan 
#đến xem có bao nhiêu người mua (1) và ko mua (0) 
print(df["buy"].value_counts())

#B2: Dọn dẹp dữ liệu
#tuổi trung bình 
average=df["age"].median()
#lương trung bình
averagesalary=df["salary"].mean()
#lấp đầy lỗ hổng(NaN)
df.fillna({"age":average},inplace=True) 
#inplace=True là sửa trực tiếp trên bảng,nếu không có sẽ tạo bản nháp mới và bảng cũ vẫn bị lỗi
df.fillna({"salary":averagesalary},inplace=True)

#kiểm tra lại dữ liệu
print(df.isnull().sum()) #đếm mỗi cột còn bao nhiêu ô NaN 
print("bảng dữ liệu đã sạch sẽ:")

#B3: trực quan hóa dữ liệu 
#tách dữ liệu thành 2 nhóm buy và notbuy
notBuy=df[df["buy"]==0]
Buy=df[df["buy"]==1]
#tạo một khung tranh kích thước 10x6 
plt.figure(figsize=(10,6))
#đưa lên trục tọa đọ (trục x:age,trục y:salary)
# notBuy(màu đỏ,hình tròn,hơi trong suốt alpha=0.7)
# Buy(màu xanh lục,hình ngôi sao hoặc tam giác(marker='^'))
plt.scatter(notBuy["age"],notBuy["salary"],color="red",label="not_buy",alpha=0.7)
plt.scatter(Buy["age"],Buy["salary"],color="green",label="buy",marker='^',s=80)#s=80 size của các chấm 

#Trang trí trông cho chuyên nghiệp 
plt.title("phân bố khách hàng mua xe theo age và salary")
plt.xlabel("age của khách hàng")
plt.ylabel("salary(triệu VNĐ)")
plt.legend()#bật bảng chú thích red là gì , green là gì...
plt.grid(True,linestyle='--',alpha=0.5)#bật lưới cho dex nhìn tọa độ 

#hiển thị
plt.show()

#B4:Chia dữ liệu 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
#Tách X,y
X=df[["age","salary"]]
y=df["buy"]

#Chia tập train,test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#chuẩn hóa dữ liệu 
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#khởi tạo ,huấn luyện model
model=LogisticRegression()
model.fit(X_train_scaled,y_train)

#chấm điểm model
from sklearn.metrics import accuracy_score

#cho model dự đoán 
y_pred=model.predict(X_test_scaled)
score=accuracy_score(y_test,y_pred)
print(f"độ chính xác là :{score*100:.2f}%\n")

#thử dự đoán mới
myage=18
mysalary=30
newclient=np.array([[myage,mysalary]])
newclient_scaled=scaler.transform(newclient)
me_pred=model.predict(newclient_scaled)
print("kết quả:",me_pred[0])  