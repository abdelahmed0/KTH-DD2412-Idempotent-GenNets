import sys
import os
import time
import numpy as np
import torch
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QCheckBox, QPushButton, QLabel, QScrollArea, QSizePolicy,
    QMessageBox,
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from tqdm import tqdm
from model.u_net_conditional import UNetConditional
from util.dataset import load_celeb_a
from util.plot_util import save_images
from util.model_util import load_checkpoint
from util.function_util import fourier_sample


class ImageGeneratorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Conditional Image Generator")
        self.setMinimumSize(1000, 800)  # Set a larger default size for better layout
        self.init_ui()
        self.normalized = True
        self.last_model = ""
        self.last_device = None
        self.image_dir = "image_gui_output/"
        os.makedirs(self.image_dir, exist_ok=True)

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        
        # Image display at the top
        self.image_label = QLabel("Generated images will appear here.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(400)  # Reserve space for a larger image
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(self.image_label)

        # Scroll area for toggles
        self.scroll_area = QScrollArea()
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)

        self.toggles = {}
        self.names = {
            '5_o_clock_shadow': True, 'arched_eyebrows': True, 'attractive': True,
            'bags_under_eyes': False, 'bald': False, 'bangs': False, 'big_lips': False,
            'big_nose': False, 'black_hair': False, 'blond_hair': False, 'blurry': False,
            'brown_hair': False, 'bushy_eyebrows': False, 'chubby': False, 'double_chin': False,
            'eyeglasses': False, 'goatee': False, 'gray_hair': False, 'heavy_makeup': False,
            'high_cheekbones': False, 'male': False, 'mouth_slightly_open': False,
            'mustache': False, 'narrow_eyes': False, 'no_beard': False, 'oval_face': False,
            'pale_skin': False, 'pointy_nose': False, 'receding_hairline': False, 
            'rosy_cheeks': False, 'sideburns': False, 'smiling': False, 'straight_hair': False,
            'wavy_hair': False, 'wearing_earrings': False, 'wearing_hat': False, 
            'wearing_lipstick': False, 'wearing_necklace': False, 'wearing_necktie': False, 
            'young': False,
        }
        for name, value in self.names.items():
            checkbox = QCheckBox(name)
            checkbox.setChecked(value)
            self.toggles[name] = checkbox
            self.scroll_layout.addWidget(checkbox)
        
        self.scroll_widget.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_widget)
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)

        # Device selection checkbox
        self.device_checkbox = QCheckBox("Use GPU if available")
        self.device_checkbox.setChecked(torch.cuda.is_available())  # Default based on CUDA availability
        self.layout.addWidget(self.device_checkbox)

        # Device selection checkbox
        self.eval_mode_checkbox = QCheckBox("Model in eval mode")
        self.eval_mode_checkbox.setChecked(True)  # Default based on CUDA availability
        self.layout.addWidget(self.eval_mode_checkbox)

        # Input box for run_id
        self.run_id_label = QLabel("Model Run ID:")
        self.run_id_input = QLineEdit()
        self.run_id_input.setPlaceholderText("Enter model run ID here (e.g., CELEBA_CONDITIONAL_UNET_2_epoch_700)")
        self.layout.addWidget(self.run_id_label)
        self.layout.addWidget(self.run_id_input)

        # Generate button at the bottom
        self.generate_button = QPushButton("Generate Images")
        self.generate_button.clicked.connect(self.generate_images)
        self.layout.addWidget(self.generate_button)


    def setup_model(self, run_id):
        device = torch.device("cuda" if self.device_checkbox.isChecked() and torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        checkpoint = load_checkpoint(f"checkpoints/{run_id}.pt")
        model = UNetConditional(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        data = load_celeb_a(batch_size=checkpoint['config']['training']['batch_size'], split='test')
        attr_names = np.array(data.dataset.attr_names)[:-1]
        return model, device, data, attr_names, checkpoint

    def get_selected_attributes(self):
        return {name for name, checkbox in self.toggles.items() if checkbox.isChecked()}

    def names_to_attributes(self, attr_names, names):
        ret = torch.tensor([[1 if attr in names or attr.lower() in names else 0 for attr in attr_names]])
        return ret if ret.sum() > 0 else None

    def generate_images(self):
        run_id = self.run_id_input.text().strip()  # Get the user-specified run_id
        if not run_id:
            QMessageBox.warning(self, "Error", "Please enter a valid model run ID!")
            return

        if self.last_model != run_id or self.last_device != self.device_checkbox.isChecked():
            self.last_model = run_id
            self.last_device = self.device_checkbox.isChecked()
            try: 
                self.model, self.device, self.data, self.attr_names, self.checkpoint = self.setup_model(run_id)
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))
                return
            
        if self.eval_mode_checkbox.isChecked():
            self.model.eval()
        else:
            self.model.train()

        selected_names = self.get_selected_attributes()
        y = self.names_to_attributes(self.attr_names, selected_names)

        n_images = 5
        n_recursions = 3
        use_fourier_sampling = self.checkpoint['config']['training']['use_fourier_sampling']
        original, reconstructed = self.generate_conditional_images(
            model=self.model, device=self.device, data=self.data, 
            n_images=n_images, n_recursions=n_recursions, 
            use_fourier_sampling=use_fourier_sampling, y=y
        )
        
        image_path = os.path.join(self.image_dir, f"output_images_{int(time.time())}.png")
        save_images(original, reconstructed, grayscale=False, normalized=self.normalized, output_path=image_path, suptitle=f'Labels: {", ".join(selected_names).strip(",")}')
        self.display_image(image_path)

    def generate_conditional_images(self, model, device, data, n_images, n_recursions, use_fourier_sampling, y):
        def get_random_batch(dataloader):
            random_idx = torch.randint(0, 50, (1,))
            for idx, batch in enumerate(dataloader):
                if idx == random_idx:
                    return batch

        batch, _ = get_random_batch(data)
        original = torch.empty(n_images, *next(iter(data))[0].shape[1:])
        reconstructed = torch.empty(n_images, n_recursions, *batch.shape[1:])
        
        with torch.inference_mode():
            for i in tqdm(range(n_images)):
                x = fourier_sample(batch) if use_fourier_sampling else torch.randn_like(batch)
                original[i] = x[0].clamp(-1.0, 1.0).cpu()
                x_hat = x[:1].to(device)
                for j in range(n_recursions):
                    x_hat = model(x_hat, y)
                    reconstructed[i, j] = x_hat[0].cpu()
        
        return original, reconstructed

    def display_image(self, image_path):
        pixmap = QPixmap(image_path)
        # Resize the QLabel to ensure the image fits and scales appropriately
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.width(), 
                self.image_label.height(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
        )


def main():
    app = QApplication(sys.argv)
    window = ImageGeneratorApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
