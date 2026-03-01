import sys
import torch
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QFrame
from PySide6.QtGui import QPixmap, QFont, QIcon
from PySide6.QtCore import Qt
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("trained_model/model.pkl", map_location=device, weights_only=False)
model.eval()

classes = ["glioma", "meningioma", "notumor", "pituitary"]

tf = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

class App(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Abbas Hilal Brain Tumor Detector")
        self.setWindowIcon(QIcon("data/icon.ico"))
        self.setMinimumSize(900,600)

        self.setStyleSheet("""
        QWidget{
            background:#0f172a;
            color:white;
            font-family:Segoe UI;
        }
        QPushButton{
            background:#2563eb;
            border-radius:10px;
            padding:12px;
            font-size:16px;
        }
        QPushButton:hover{
            background:#1d4ed8;
        }
        QLabel{
            font-size:16px;
        }
        """)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedHeight(350)
        self.image_label.setFrameShape(QFrame.Box)

        self.result_label = QLabel("Upload MRI Image")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Segoe UI",18))

        self.upload_btn = QPushButton("Upload MRI Image")
        self.upload_btn.clicked.connect(self.load_image)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.upload_btn)

        self.setLayout(layout)

    def load_image(self):
        file,_ = QFileDialog.getOpenFileName(self,"Open Image","","Images (*.png *.jpg *.jpeg)")
        if not file:
            return

        pix = QPixmap(file).scaled(400,400,Qt.KeepAspectRatio)
        self.image_label.setPixmap(pix)

        img = Image.open(file).convert("RGB")
        tensor = tf(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            pred = logits.argmax(1).item()

        result = classes[pred]

        if result == "notumor":
            self.result_label.setText("Result: No Tumor Detected")
        else:
            self.result_label.setText(f"Result: {result} Tumor Detected")

app = QApplication(sys.argv)
window = App()
window.show()
sys.exit(app.exec())