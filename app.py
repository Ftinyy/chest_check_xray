# 导入必要的库
import streamlit as st
import torch
from torch import nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 定义模型类
class NeuralNetwork(nn.Module):
    def __init__(self, dropout_p=0.3):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(256 * 256, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(p=dropout_p)

        self.fc2 = nn.Linear(512, 600)
        self.bn2 = nn.BatchNorm1d(600)
        self.dropout2 = nn.Dropout(p=dropout_p)

        self.fc3 = nn.Linear(600, 200)
        self.bn3 = nn.BatchNorm1d(200)
        self.dropout3 = nn.Dropout(p=dropout_p)

        self.fc4 = nn.Linear(200, 2)

    def forward(self, x):
        x = x.view(-1, 256 * 256)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.sigmoid(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.sigmoid(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

# 加载训练好的模型权重
def load_model():
    model = NeuralNetwork()
    # 确保在正确的设备上加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load('fxy_chest.pth', map_location=device))
    model.eval()
    return model

# 图像预处理函数
def preprocess_image(image):
    # 转换为灰度图像
    image = image.convert('L')
    # 调整大小为 256x256
    image = image.resize((256, 256))
    # 转换为 numpy 数组
    image_array = np.array(image)
    # 归一化到 [0, 1] 范围
    image_array = image_array / 255.0
    # 添加通道维度
    image_array = image_array[np.newaxis, :, :]
    # 转换为 PyTorch 张量
    image_tensor = torch.tensor(image_array, dtype=torch.float32)
    return image_tensor

# 预测函数
def predict(image_tensor, model):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    return predicted_class, confidence

# 批量预测函数
def batch_predict(image_tensors, model):
    with torch.no_grad():
        outputs = model(image_tensors)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1).numpy()
        confidences = probabilities.numpy()
    return predicted_classes, confidences

# Streamlit 界面
def main():
    st.title("胸部X光图像分类")
    st.sidebar.title("功能选择")
    app_mode = st.sidebar.selectbox("选择功能", ["单张图像预测", "批量预测", "模型性能", "数据集可视化", "数据集统计"])

    # 加载模型
    model = load_model()

    if app_mode == "单张图像预测":
        st.write("上传一张胸部X光图像，模型将预测它是正常还是异常。")
        uploaded_file = st.file_uploader("选择一张图像...", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='上传的图像', use_column_width=True)
            image_tensor = preprocess_image(image)
            if st.button('预测'):
                predicted_class, confidence = predict(image_tensor, model)
                class_names = ['正常', '异常']
                st.write(f"预测结果: {class_names[predicted_class]}")
                st.write(f"置信度: {confidence:.2%}")

    elif app_mode == "批量预测":
        st.write("上传多张胸部X光图像，模型将批量预测它们是正常还是异常。")
        uploaded_files = st.file_uploader("选择多张图像...", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        if uploaded_files:
            images = []
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                images.append(image)
            st.write(f"共上传了 {len(images)} 张图像")
            image_tensors = [preprocess_image(img) for img in images]
            image_tensors = torch.stack(image_tensors)
            if st.button('批量预测'):
                predicted_classes, confidences = batch_predict(image_tensors, model)
                class_names = ['正常', '异常']
                results = []
                for i, img in enumerate(images):
                    st.image(img, caption=f'预测结果: {class_names[predicted_classes[i]]} (置信度: {confidences[i][predicted_classes[i]]:.2%})', use_column_width=True)
                    results.append({
                        "图像": img,
                        "预测结果": class_names[predicted_classes[i]],
                        "置信度": confidences[i][predicted_classes[i]]
                    })
                st.write(pd.DataFrame(results))


    elif app_mode == "模型性能":
        st.write("模型在训练集、验证集和测试集上的性能指标")
        # 示例：训练与验证准确率和损失
        epochs = list(range(1, 11))
        train_acc = [0.80, 0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]
        val_acc = [0.78, 0.82, 0.85, 0.87, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94]
        train_loss = [0.6, 0.5, 0.42, 0.35, 0.30, 0.28, 0.25, 0.23, 0.21, 0.20]
        val_loss = [0.65, 0.55, 0.45, 0.40, 0.38, 0.36, 0.34, 0.32, 0.30, 0.29]
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].plot(epochs, train_loss, label='train_loss')
        ax[0].plot(epochs, val_loss, label='vaild_loss')
        ax[0].set_title("loss")
        ax[0].legend()
        ax[1].plot(epochs, train_acc, label='train_accuracy')
        ax[1].plot(epochs, val_acc, label='valid_accuracy')
        ax[1].set_title("accuracy")
        ax[1].legend()
        st.pyplot(fig)



    elif app_mode == "数据集可视化":
        st.write("展示部分数据集样本")
        sample_dir = r"D:\CV\Chest\IU-Xray\images\images_normalized"
        sample_images = os.listdir(sample_dir)[:10]
        for img_name in sample_images:
            img = Image.open(os.path.join(sample_dir, img_name))
            st.image(img, caption=img_name, width=200)


    elif app_mode == "数据集统计":
        st.write("显示正常与异常图像数量的统计")
        # 假设你有标签数据
        data = pd.read_csv(r"D:\CV\Chest\IU-Xray\path_label_process.csv")  # 包含列：['filename', 'label']
        count = data['labels'].value_counts()
        st.bar_chart(count)


if __name__ == "__main__":
    main()