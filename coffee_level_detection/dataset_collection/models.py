
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
from pydantic import BaseModel
from datetime import datetime
class annotation(BaseModel):
    filename: str
    coffee_level: int
    timestamp: datetime
    annotator: str
    version: str
    
class CoffeeLevelSelector:
    """Interactive coffee level selector using matplotlib with keyboard navigation."""
    
    def __init__(self, img, filename=""):
        self.img = img
        self.filename = filename
        self.current_level = 0  # Start at 0 cups
        self.max_level = 10     # Maximum 10 cups
        self.selected = False   # Whether user has made selection
        self.exit_requested = False
        
        # Setup matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # Display the image
        if len(img.shape) == 3:
            # Convert BGR to RGB for matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        self.ax.imshow(img_rgb)
        self.ax.set_title(f"Coffee Level Selection - {filename}")
        
        # Create text overlay for instructions and current selection
        self.instruction_text = self.fig.text(0.02, 0.95, self._get_instruction_text(), 
                                            fontsize=12, color='white', 
                                            bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        self.level_text = self.fig.text(0.02, 0.85, self._get_level_text(), 
                                       fontsize=16, color='yellow', weight='bold',
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
        
        # Remove axes for cleaner look
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        plt.tight_layout()
        
    def _get_instruction_text(self):
        return ("Instructions:\n"
                "← → : Change coffee level (0-10)\n"
                "q   : Save selection and continue\n"
                "esc : Skip this image")
    
    def _get_level_text(self):
        if self.current_level == 0:
            return f"Current Selection: {self.current_level} cups (Empty)"
        elif self.current_level == 1:
            return f"Current Selection: {self.current_level} cup"
        else:
            return f"Current Selection: {self.current_level} cups"
    
    def _on_key_press(self, event):
        """Handle keyboard input for navigation and selection."""
        if event.key == 'left':
            # Move to previous level (with wrapping)
            self.current_level = (self.current_level - 1) % (self.max_level + 1)
            self._update_display()
            
        elif event.key == 'right':
            # Move to next level (with wrapping)
            self.current_level = (self.current_level + 1) % (self.max_level + 1)
            self._update_display()
            
        elif event.key == 'q':
            # Save selection and close
            self.selected = True
            plt.close(self.fig)
            
        elif event.key == 'escape':
            # Skip this image
            self.exit_requested = True
            plt.close(self.fig)
    
    def _update_display(self):
        """Update the level text display."""
        self.level_text.set_text(self._get_level_text())
        self.fig.canvas.draw()
    
    def show(self):
        """Display the selector and wait for user input."""
        plt.show()
        return self.current_level if self.selected else None

class ZeroCoffeeLevelRelabeler:
        """
        Interactive relabeler for coffee_level==0 images.
        Lets user choose between None (no coffee pot) and 0 (coffee pot mostly empty).
        """
        def __init__(self, img, filename=""):
            self.img = img
            self.filename = filename
            self.selection = None  # None or 0
            self.selected = False
            self.exit_requested = False
            self.options = [None, 0]
            self.option_names = ["None (no coffee pot)", "0 (coffee pot mostly empty)"]
            self.current_idx = 0
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
            self.ax.imshow(img_rgb)
            self.ax.set_title(f"Relabel Zero Coffee Level - {filename}")
            self.instruction_text = self.fig.text(0.02, 0.95, self._get_instruction_text(),
                                                 fontsize=12, color='white',
                                                 bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
            self.level_text = self.fig.text(0.02, 0.85, self._get_level_text(),
                                           fontsize=16, color='yellow', weight='bold',
                                           bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            plt.tight_layout()

        def _get_instruction_text(self):
            return ("Instructions:\n"
                    "← → : Change selection (None/0)\n"
                    "q   : Save selection and continue\n"
                    "esc : Skip this image")

        def _get_level_text(self):
            return f"Current Selection: {self.option_names[self.current_idx]}"

        def _on_key_press(self, event):
            if event.key == 'left':
                self.current_idx = (self.current_idx - 1) % len(self.options)
                self._update_display()
            elif event.key == 'right':
                self.current_idx = (self.current_idx + 1) % len(self.options)
                self._update_display()
            elif event.key == 'q':
                self.selection = self.options[self.current_idx]
                self.selected = True
                plt.close(self.fig)
            elif event.key == 'escape':
                self.exit_requested = True
                plt.close(self.fig)

        def _update_display(self):
            self.level_text.set_text(self._get_level_text())
            self.fig.canvas.draw()

        def show(self):
            plt.show()
            return self.selection if self.selected else None


class MultiPolygonMaskBuilder:
    """Legacy polygon mask builder - kept for compatibility."""
    
    def __init__(self, img):
        self.img = img
        self.mask = np.zeros(img.shape[:2], dtype=np.uint8)
        self.exit_requested = False
        
        # For backward compatibility, we'll create a simple interface
        print("MultiPolygonMaskBuilder is deprecated. Use CoffeeLevelSelector instead.")
        
        # Create a basic matplotlib interface
        fig, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img)
        ax.set_title("Press 'q' to continue (legacy mask builder)")
        
        def on_key(event):
            if event.key == 'q':
                plt.close(fig)
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()