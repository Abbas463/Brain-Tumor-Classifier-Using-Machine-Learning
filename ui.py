import sys
import torch
import json
import os
import platform
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, 
                              QFileDialog, QVBoxLayout, QHBoxLayout, QFrame,
                              QScrollArea, QFrame, QProgressBar)
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QSettings
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

class ThemeManager:
    DARK_THEME = {
        'background': 'qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0a0e27, stop:0.5 #1a1f3a, stop:1 #0f172a)',
        'text_color': '#ffffff',
        'title_color': '#ffffff',
        'subtitle_color': '#94a3b8',
        'status_color': '#64b5f6',
        'card_bg': 'rgba(255, 255, 255, 0.05)',
        'card_border': 'rgba(255, 255, 255, 0.1)',
        'button_bg': 'rgba(255, 255, 255, 0.1)',
        'button_border': 'rgba(255, 255, 255, 0.2)',
        'button_hover': 'rgba(255, 255, 255, 0.2)',
        'button_pressed': 'rgba(255, 255, 255, 0.15)',
        'scrollbar_bg': 'rgba(255, 255, 255, 0.1)',
        'scrollbar_handle': 'rgba(255, 255, 255, 0.3)',
        'scrollbar_hover': 'rgba(255, 255, 255, 0.5)',
        'progress_bg': 'rgba(255, 255, 255, 0.1)',
        'progress_chunk': 'qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4fc3f7, stop:0.5 #29b6f6, stop:1 #0288d1)',
        'success_bg': 'rgba(76, 175, 80, 0.2)',
        'success_border': 'rgba(76, 175, 80, 0.5)',
        'success_color': '#4caf50',
        'error_bg': 'rgba(244, 67, 54, 0.2)',
        'error_border': 'rgba(244, 67, 54, 0.5)',
        'error_color': '#f44336',
        'image_placeholder_bg': 'rgba(0, 0, 0, 0.3)',
        'image_border': 'rgba(255, 255, 255, 0.2)',
        'footer_color': '#64748b'
    }
    
    LIGHT_THEME = {
        'background': 'qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f8fafc, stop:0.5 #e2e8f0, stop:1 #f1f5f9)',
        'text_color': '#1e293b',
        'title_color': '#0f172a',
        'subtitle_color': '#64748b',
        'status_color': '#0369a1',
        'card_bg': 'rgba(255, 255, 255, 0.8)',
        'card_border': 'rgba(0, 0, 0, 0.1)',
        'button_bg': 'rgba(255, 255, 255, 0.7)',
        'button_border': 'rgba(0, 0, 0, 0.2)',
        'button_hover': 'rgba(255, 255, 255, 0.9)',
        'button_pressed': 'rgba(255, 255, 255, 0.8)',
        'scrollbar_bg': 'rgba(0, 0, 0, 0.1)',
        'scrollbar_handle': 'rgba(0, 0, 0, 0.3)',
        'scrollbar_hover': 'rgba(0, 0, 0, 0.5)',
        'progress_bg': 'rgba(0, 0, 0, 0.1)',
        'progress_chunk': 'qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0ea5e9, stop:0.5 #0284c7, stop:1 #0369a1)',
        'success_bg': 'rgba(34, 197, 94, 0.2)',
        'success_border': 'rgba(34, 197, 94, 0.5)',
        'success_color': '#16a34a',
        'error_bg': 'rgba(239, 68, 68, 0.2)',
        'error_border': 'rgba(239, 68, 68, 0.5)',
        'error_color': '#dc2626',
        'image_placeholder_bg': 'rgba(0, 0, 0, 0.05)',
        'image_border': 'rgba(0, 0, 0, 0.2)',
        'footer_color': '#64748b'
    }
    
    def __init__(self):
        self.settings = QSettings('BrainTumorClassifier', 'ThemeSettings')
        saved_theme = self.settings.value('theme')
        if saved_theme:
            self.current_theme = saved_theme
        else:
            self.current_theme = self.detect_system_theme()
            self.settings.setValue('theme', self.current_theme)
        
    def detect_system_theme(self):
        """Detect system theme preference"""
        try:
            if platform.system() == "Windows":
                import winreg
                try:
                    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                                       r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as key:
                        apps_use_light_theme, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
                        return 'light' if apps_use_light_theme else 'dark'
                except (FileNotFoundError, OSError):
                    return 'dark'
            else:
                return 'dark'
        except Exception:
            return 'dark'
        
    def get_current_theme(self):
        return self.DARK_THEME if self.current_theme == 'dark' else self.LIGHT_THEME
    
    def toggle_theme(self):
        self.current_theme = 'light' if self.current_theme == 'dark' else 'dark'
        self.settings.setValue('theme', self.current_theme)
        return self.get_current_theme()
    
    def set_theme(self, theme_name):
        if theme_name in ['dark', 'light']:
            self.current_theme = theme_name
            self.settings.setValue('theme', self.current_theme)
            return self.get_current_theme()
        return None

class ThemeToggleButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.theme_manager = getattr(parent, 'theme_manager', None) or ThemeManager()
        self.update_style()
        self.update_icon()
        
    def update_style(self):
        theme = self.theme_manager.get_current_theme()
        self.setStyleSheet(f"""
            QPushButton {{
                background: {theme['button_bg']};
                border: 2px solid {theme['button_border']};
                border-radius: 15px;
                padding: 8px 15px;
                font-size: 20px;
                font-weight: 600;
                color: {theme['text_color']};
                min-width: 50px;
                min-height: 35px;
                max-width: 60px;
                max-height: 40px;
            }}
            QPushButton:hover {{
                background: {theme['button_hover']};
                border: 2px solid {theme['button_border']};
            }}
            QPushButton:pressed {{
                background: {theme['button_pressed']};
            }}
        """)
        
    def update_icon(self):
        icon = "🌙" if self.theme_manager.current_theme == 'light' else "☀️"
        self.setText(icon)
        self.setToolTip("Switch to " + ("Light Mode" if self.theme_manager.current_theme == 'dark' else "Dark Mode"))

class GlassButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.theme_manager = getattr(parent, 'theme_manager', None) or ThemeManager()
        self.update_style()
        
    def update_style(self):
        theme = self.theme_manager.get_current_theme()
        self.setStyleSheet(f"""
            QPushButton {{
                background: {theme['button_bg']};
                border: 2px solid {theme['button_border']};
                border-radius: 15px;
                padding: 15px 30px;
                font-size: 16px;
                font-weight: 600;
                color: {theme['text_color']};
            }}
            QPushButton:hover {{
                background: {theme['button_hover']};
                border: 2px solid {theme['button_border']};
            }}
            QPushButton:pressed {{
                background: {theme['button_pressed']};
            }}
        """)
        
class ModernCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.theme_manager = getattr(parent, 'theme_manager', None) or ThemeManager()
        self.update_style()
        
    def update_style(self):
        theme = self.theme_manager.get_current_theme()
        self.setStyleSheet(f"""
            QFrame {{
                background: {theme['card_bg']};
                border: 1px solid {theme['card_border']};
                border-radius: 20px;
            }}
        """)

class App(QWidget):
    def __init__(self):
        super().__init__()
        
        self.theme_manager = ThemeManager()
        
        self.setWindowTitle("Abbas Hilal AI - Brain Tumor Detection System")
        self.setWindowIcon(QIcon("data/icon.ico"))
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinMaxButtonsHint)
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        self.setAcceptDrops(True)
        
        self.is_custom_maximized = False
        
        self.update_theme_style()
        
        self.setup_ui()
        self.setup_animations()
        self.showMaximized()
        
    def update_theme_style(self):
        theme = self.theme_manager.get_current_theme()
        self.setStyleSheet(f"""
            QWidget {{
                background: {theme['background']};
                color: {theme['text_color']};
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }}
            
            QLabel#title {{
                font-size: 32px;
                font-weight: 700;
                color: {theme['title_color']};
                padding: 20px;
            }}
            
            QLabel#subtitle {{
                font-size: 14px;
                color: {theme['subtitle_color']};
                padding: 10px;
            }}
            
            QLabel#result {{
                font-size: 24px;
                font-weight: 600;
                padding: 20px;
                background: {theme['card_bg']};
                border-radius: 15px;
                margin: 10px;
                color: {theme['text_color']};
            }}
            
            QLabel#status {{
                font-size: 16px;
                color: {theme['status_color']};
                padding: 10px;
            }}
            
            QProgressBar {{
                border: none;
                border-radius: 10px;
                text-align: center;
                color: {theme['text_color']};
                background: {theme['progress_bg']};
            }}
            
            QProgressBar::chunk {{
                background: {theme['progress_chunk']};
                border-radius: 10px;
            }}
            
            QLabel#footer {{
                color: {theme['footer_color']};
                font-size: 12px;
                padding: 10px;
            }}
        """)
        
        if hasattr(self, 'scroll_area'):
            self.update_scrollbar_style()
            
        if hasattr(self, 'image_label'):
            self.update_image_label_style()
            
        self.update_widgets_style()
            
    def update_scrollbar_style(self):
        theme = self.theme_manager.get_current_theme()
        self.scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background: transparent;
            }}
            QScrollBar:vertical {{
                background: {theme['scrollbar_bg']};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background: {theme['scrollbar_handle']};
                border-radius: 6px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {theme['scrollbar_hover']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar:horizontal {{
                background: {theme['scrollbar_bg']};
                height: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:horizontal {{
                background: {theme['scrollbar_handle']};
                border-radius: 6px;
                min-width: 20px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background: {theme['scrollbar_hover']};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
        """)
        
    def update_image_label_style(self):
        theme = self.theme_manager.get_current_theme()
        self.image_label.setStyleSheet(f"""
            QLabel {{
                background: {theme['image_placeholder_bg']};
                border: 2px dashed {theme['image_border']};
                border-radius: 15px;
                padding: 20px;
            }}
        """)
        
    def update_widgets_style(self):
        for widget in self.findChildren(GlassButton):
            widget.theme_manager = self.theme_manager
            widget.update_style()
            
        for widget in self.findChildren(ThemeToggleButton):
            widget.theme_manager = self.theme_manager
            widget.update_style()
            widget.update_icon()
            
        for widget in self.findChildren(ModernCard):
            widget.theme_manager = self.theme_manager
            widget.update_style()
            
    def toggle_theme(self):
        self.theme_manager.toggle_theme()
        self.update_theme_style()
        
        if hasattr(self, 'theme_toggle_btn'):
            self.theme_toggle_btn.update_icon()
        
    def setup_ui(self):
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        content_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(30)
        
        header_widget = ModernCard()
        header_layout = QVBoxLayout()
        
        title_row = QHBoxLayout()
        
        title_label = QLabel("Abbas Hilal AI")
        title_label.setObjectName("title")
        title_label.setAlignment(Qt.AlignLeft)
        
        self.theme_toggle_btn = ThemeToggleButton(self)
        self.theme_toggle_btn.clicked.connect(self.toggle_theme)
        
        title_row.addWidget(title_label)
        title_row.addStretch()
        title_row.addWidget(self.theme_toggle_btn)
        
        subtitle_label = QLabel("Advanced Brain Tumor Detection System By Afghan Boy")
        subtitle_label.setObjectName("subtitle")
        subtitle_label.setAlignment(Qt.AlignCenter)
        
        header_layout.addLayout(title_row)
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
        
        footer_label = QLabel("Powered Afghan Boy")
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setObjectName("footer")
        
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
                theme = self.theme_manager.get_current_theme()
                self.result_label.setStyleSheet(f"""
                    QLabel#result {{
                        background: {theme['success_bg']};
                        border: 2px solid {theme['success_border']};
                        color: {theme['success_color']};
                    }}
                """)
                self.status_label.setText("Analysis complete - No abnormalities detected")
            else:
                tumor_type = result.replace('_', ' ').title()
                self.result_label.setText(
                    f"⚠️ {tumor_type} Tumor Detected\n"
                    f"Confidence: {confidence:.1f}%\n"
                    f"⚡ Immediate medical consultation recommended"
                )
                theme = self.theme_manager.get_current_theme()
                self.result_label.setStyleSheet(f"""
                    QLabel#result {{
                        background: {theme['error_bg']};
                        border: 2px solid {theme['error_border']};
                        color: {theme['error_color']};
                    }}
                """)
                self.status_label.setText(f"Analysis complete - {tumor_type} detected")
                
        except Exception as e:
            self.progress_bar.setVisible(False)
            self.result_label.setText(f"❌ Analysis Error\n{str(e)}")
            theme = self.theme_manager.get_current_theme()
            self.result_label.setStyleSheet(f"""
                QLabel#result {{
                    background: {theme['error_bg']};
                    border: 2px solid {theme['error_border']};
                    color: {theme['error_color']};
                }}
            """)
            self.status_label.setText("Error during analysis")
            
    def clear_image(self):
        self.image_label.clear()
        self.image_label.setText("📷\n\nDrop MRI image here\nor click upload button")
        self.result_label.setText("Awaiting Image Analysis")
        theme = self.theme_manager.get_current_theme()
        self.result_label.setStyleSheet(f"""
            QLabel#result {{
                background: {theme['card_bg']};
                border: none;
                color: {theme['text_color']};
            }}
        """)
        self.status_label.setText("Ready to analyze")
        self.analyze_btn.setEnabled(False)
        if hasattr(self, 'current_image_path'):
            delattr(self, 'current_image_path')

app = QApplication(sys.argv)
window = App()
window.show()
sys.exit(app.exec())