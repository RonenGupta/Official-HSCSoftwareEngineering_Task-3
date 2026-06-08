from pathlib import Path
import os
from PIL import Image
import re

class SecurityManager():
    """Handles dataset path validation"""
    def __init__(self, path):
        # Store the user-provided dataset path
        self.path = path
        # Base directory where datasets are allowed to exist.
        # ALL validated paths must be inside this directory.
        self.base_data_dir = "./datasets"

    def validate_path(self):
        """Validates the dataset path entered by a user"""

        # 1. Checks if the path actually exists
        if not self.path or not isinstance(self.path, str):
            raise ValueError("Path must be a non-empty string")
        
        # 2. Alters the path into its resolute form
        try:
            resolved = Path(self.path).expanduser().resolve(strict = False)
        except Exception as e:
            raise ValueError(f"Invalid Path: {e}")
        
        resolved_str = str(resolved)

        # 3. Resolves the allowed base, checks if the path is inside the allowed base
        allowed_base = Path(self.base_data_dir).resolve()
        if not resolved_str.startswith(str(allowed_base) + os.sep) and resolved != allowed_base:
            raise ValueError(f"Path must be inside the allowed directory: {self.base_data_dir}")
        
        # 4. Creates the train and test directory according to path structure
        train_dir = resolved / "train" 
        test_dir = resolved / "test"

        # 5. Checks if train directory exists and is a directory
        if not train_dir.exists() or not train_dir.is_dir():
            raise ValueError("Train directory must exist inside the dataset folder")

        # 6. Checks if test directory exists and is a directory
        if not test_dir.exists() or not test_dir.is_dir():
            raise ValueError("Test directory must exist inside the dataset folder")
        
        # 7. List comprehension used to get the class names from the train and test directory by iterating through it 
        # and checking if it is a directory
        train_classes = [d for d in train_dir.iterdir() if d.is_dir()]
        test_classes = [d for d in test_dir.iterdir() if d.is_dir()]

        # 8. If there are less than two classes in either train or test folder, raise an error
        if len(train_classes) < 2 or len(test_classes) < 2:
            raise ValueError("Train and Test folder must have greater than 2 classes")

        # 9. Get the actual class names in the train and test directory through list comprehension and sorting
        train_class_names = sorted([d.name for d in train_classes])
        test_class_names = sorted([d.name for d in test_classes])

        # 10. If the class names in train folder do not match test folder, raise an error
        if train_class_names != test_class_names:
            raise ValueError("Train and test folder must have the same classes")
        
        # 11. Block spaces, unicode, emojis, slashes and hidden folders by using a regex expression
        VALID_NAME = re.compile(r"^[A-Za-z0-9_\-]+$")

        for name in train_class_names:
            if not VALID_NAME.match(name):
                raise ValueError(f"Invalid class name: {name}")
        
        for name in test_class_names:
            if not VALID_NAME.match(name):
                raise ValueError(f"Invalid class name: {name}")
        
        # 12. Iterates through every file in the train and test folder and stores it in images
        all_images = []
        for class_dir in train_classes + test_classes:
            all_images.extend(class_dir.glob("*"))
        
        # 13. Check for allowed image file extensions 
        ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

        for f in all_images:
            if f.suffix.lower() not in ALLOWED_EXT:
                raise ValueError(f"Invalid file extension: {f}")
        
        # 14. Check if less than 5 images in a class folder
        MIN_IMAGES_PER_CLASS = 5

        for class_dir in train_classes + test_classes:
            count = len(list(class_dir.glob("*")))
            if count < MIN_IMAGES_PER_CLASS:
                raise ValueError(f"Not enough images in {class_dir.name}: {count}")
            
        # 15. Check if more than 5000 images in a class folder
        MAX_IMAGES_PER_CLASS = 5000

        for class_dir in train_classes + test_classes:
            count = len(list(class_dir.glob("*")))
            if count > MAX_IMAGES_PER_CLASS:
                raise ValueError(f"Too many images in {class_dir.name}: {count}")
        
        # 16. Prevent users from loading datasets at 500 MB or above
        MAX_DATASET_SIZE_MB = 500
        
        total_size = sum(f.stat().st_size for f in all_images) / (1024 * 1024)
        if total_size > MAX_DATASET_SIZE_MB:
            raise ValueError(f"Dataset too large ({total_size: .2f} MB). Max allowed is {MAX_DATASET_SIZE_MB} MB")
            
        # 17. Function verifies if it is an image or not
        def verify_image(path):
            try:
                with Image.open(path) as img:
                    img.verify()
                return True
            except Exception:
                return False 


        # 18. Stores all files which are not images
        invalid_images = [f for f in all_images if not verify_image(f)]

        # 19. If there are one or more files which are not images, raise an error
        if len(invalid_images) > 0:
            raise ValueError(f"Train or test folder contain one or more files that are not an image: {invalid_images}")
        # 20. Return true if the path has been validated and no errors have been raised
        return True
           