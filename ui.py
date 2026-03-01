import sys
import torch
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, 
                              QFileDialog, QVBoxLayout, QHBoxLayout, QFrame,
                              QScrollArea, QFrame, QProgressBar)
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QEvent, QMimeData
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

class GlassButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 0.1);
                border: 2px solid rgba(255, 255, 255, 0.2);
                border-radius: 15px;
                padding: 15px 30px;
                font-size: 16px;
                font-weight: 600;
                color: white;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.2);
                border: 2px solid rgba(255, 255, 255, 0.3);
            }
            QPushButton:pressed {
                background: rgba(255, 255, 255, 0.15);
            }
        """)

class ModernCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 20px;
            }
        """)

class App(QWidget):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Abbas Hilal AI - Brain Tumor Detection System")
        self.setWindowIcon(QIcon("data/icon.ico"))
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinMaxButtonsHint)
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        self.setAcceptDrops(True)
        
        self.is_custom_maximized = False
        
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0a0e27, stop:0.5 #1a1f3a, stop:1 #0f172a);
                color: #ffffff;
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }
            
            QLabel#title {
                font-size: 32px;
                font-weight: 700;
                color: #ffffff;
                padding: 20px;
            }
            
            QLabel#subtitle {
                font-size: 14px;
                color: #94a3b8;
                padding: 10px;
            }
            
            QLabel#result {
                font-size: 24px;
                font-weight: 600;
                padding: 20px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 15px;
                margin: 10px;
            }
            
            QLabel#status {
                font-size: 16px;
                color: #64b5f6;
                padding: 10px;
            }
            
            QProgressBar {
                border: none;
                border-radius: 10px;
                text-align: center;
                color: white;
                background: rgba(255, 255, 255, 0.1);
            }
            
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4fc3f7, stop:0.5 #29b6f6, stop:1 #0288d1);
                border-radius: 10px;
            }
        """)
        
        self.setup_ui()
        self.setup_animations()
        self.showMaximized()
        
    def setup_ui(self):
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: rgba(255, 255, 255, 0.1);
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 0.3);
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(255, 255, 255, 0.5);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background: rgba(255, 255, 255, 0.1);
                height: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background: rgba(255, 255, 255, 0.3);
                border-radius: 6px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background: rgba(255, 255, 255, 0.5);
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)
        
        content_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(30)
        
        header_widget = ModernCard()
        header_layout = QVBoxLayout()
        
        title_label = QLabel("Abbas Hilal AI")
        title_label.setObjectName("title")
        title_label.setAlignment(Qt.AlignCenter)
        
        subtitle_label = QLabel("Advanced Brain Tumor Detection System By Afghan Boy")
        subtitle_label.setObjectName("subtitle")
        subtitle_label.setAlignment(Qt.AlignCenter)
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        header_widget.setLayout(header_layout)
        
        content_layout = QHBoxLayout()
        content_layout.setSpacing(30)
        
        left_panel = ModernCard()
        left_layout = QVBoxLayout()
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setMaximumSize(600, 600)
        self.image_label.setStyleSheet("""
            QLabel {
                background: rgba(0, 0, 0, 0.3);
                border: 2px dashed rgba(255, 255, 255, 0.2);
                border-radius: 15px;
                padding: 20px;
            }
        """)
        self.image_label.setText("📷\n\nDrop MRI image here\nor click upload button")
        self.image_label.setWordWrap(True)
        
        left_layout.addWidget(self.image_label)
        left_panel.setLayout(left_layout)
        
        right_panel = ModernCard()
        right_layout = QVBoxLayout()
        right_layout.setSpacing(20)
        
        self.upload_btn = GlassButton("📤 Upload MRI Scan")
        self.upload_btn.setMinimumHeight(60)
        self.upload_btn.clicked.connect(self.load_image)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        self.status_label = QLabel("Ready to analyze")
        self.status_label.setObjectName("status")
        self.status_label.setAlignment(Qt.AlignCenter)
        
        self.result_label = QLabel("Awaiting Image Analysis")
        self.result_label.setObjectName("result")
        self.result_label.setAlignment(Qt.AlignCenter)
        
        button_layout = QHBoxLayout()
        
        self.clear_btn = GlassButton("🗑️ Clear")
        self.clear_btn.clicked.connect(self.clear_image)
        
        self.analyze_btn = GlassButton("🔍 Analyze")
        self.analyze_btn.clicked.connect(self.analyze_image)
        self.analyze_btn.setEnabled(False)
        
        button_layout.addWidget(self.clear_btn)
        button_layout.addWidget(self.analyze_btn)
        
        right_layout.addWidget(self.upload_btn)
        right_layout.addWidget(self.progress_bar)
        right_layout.addWidget(self.status_label)
        right_layout.addWidget(self.result_label)
        right_layout.addLayout(button_layout)
        right_layout.addStretch()
        
        right_panel.setLayout(right_layout)
        
        content_layout.addWidget(left_panel, 2)
        content_layout.addWidget(right_panel, 1)
        
        footer_label = QLabel("Powered by Deep Learning • Medical Imaging AI")
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setStyleSheet("""
            QLabel {
                color: #64748b;
                font-size: 12px;
                padding: 10px;
            }
        """)
        
        main_layout.addWidget(header_widget)
        main_layout.addLayout(content_layout)
        main_layout.addWidget(footer_label)
        
        content_widget.setLayout(main_layout)
        
        self.scroll_area.setWidget(content_widget)
        
        window_layout = QVBoxLayout()
        window_layout.setContentsMargins(0, 0, 0, 0)
        window_layout.addWidget(self.scroll_area)
        
        self.setLayout(window_layout)
        
    def setup_animations(self):
        self.fade_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_animation.setDuration(1000)
        self.fade_animation.setStartValue(0.0)
        self.fade_animation.setEndValue(1.0)
        self.fade_animation.setEasingCurve(QEasingCurve.InOutCubic)
        self.fade_animation.start()
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            self.load_image_from_path(files[0])
    
    def load_image_from_path(self, file_path):
        pixmap = QPixmap(file_path)
        label_size = self.image_label.size()
        scaled_pixmap = pixmap.scaled(
            label_size, 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setScaledContents(True)
        
        self.analyze_btn.setEnabled(True)
        self.status_label.setText("Image loaded - Ready for analysis")
        self.current_image_path = file_path
        
    def load_image(self):
        file, _ = QFileDialog.getOpenFileName(
            self, 
            "Select MRI Image", 
            "", 
            "Medical Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if not file:
            return
            
        pixmap = QPixmap(file)
        label_size = self.image_label.size()
        scaled_pixmap = pixmap.scaled(
            label_size, 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setScaledContents(True)
        
        self.analyze_btn.setEnabled(True)
        self.status_label.setText("Image loaded - Ready for analysis")
        self.current_image_path = file
        
    def analyze_image(self):
        if not hasattr(self, 'current_image_path'):
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("Analyzing image...")
        self.result_label.setText("Processing...")
        
        QTimer.singleShot(100, self._perform_analysis)
        
    def _perform_analysis(self):
        try:
            img = Image.open(self.current_image_path).convert("RGB")
            tensor = tf(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(tensor)
                probabilities = torch.softmax(logits, dim=1)
                pred = logits.argmax(1).item()
                confidence = probabilities[0][pred].item() * 100
                
            result = classes[pred]
            
            self.progress_bar.setVisible(False)
            
            if result == "notumor":
                self.result_label.setText(
                    f"✅ No Tumor Detected\n"
                    f"Confidence: {confidence:.1f}%"
                )
                self.result_label.setStyleSheet("""
                    QLabel#result {
                        background: rgba(76, 175, 80, 0.2);
                        border: 2px solid rgba(76, 175, 80, 0.5);
                        color: #4caf50;
                    }
                """)
                self.status_label.setText("Analysis complete - No abnormalities detected")
            else:
                tumor_type = result.replace('_', ' ').title()
                self.result_label.setText(
                    f"⚠️ {tumor_type} Tumor Detected\n"
                    f"Confidence: {confidence:.1f}%\n"
                    f"⚡ Immediate medical consultation recommended"
                )
                self.result_label.setStyleSheet("""
                    QLabel#result {
                        background: rgba(244, 67, 54, 0.2);
                        border: 2px solid rgba(244, 67, 54, 0.5);
                        color: #f44336;
                    }
                """)
                self.status_label.setText(f"Analysis complete - {tumor_type} detected")
                
        except Exception as e:
            self.progress_bar.setVisible(False)
            self.result_label.setText(f"❌ Analysis Error\n{str(e)}")
            self.status_label.setText("Error during analysis")
            
    def clear_image(self):
        self.image_label.clear()
        self.image_label.setText("📷\n\nDrop MRI image here\nor click upload button")
        self.result_label.setText("Awaiting Image Analysis")
        self.result_label.setStyleSheet("""
            QLabel#result {
                background: rgba(255, 255, 255, 0.05);
                border: none;
                color: white;
            }
        """)
        self.status_label.setText("Ready to analyze")
        self.analyze_btn.setEnabled(False)
        if hasattr(self, 'current_image_path'):
            delattr(self, 'current_image_path')

app = QApplication(sys.argv)
window = App()
window.show()
sys.exit(app.exec())