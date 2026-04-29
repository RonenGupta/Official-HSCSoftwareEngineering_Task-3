from pathlib import Path
import os
import imghdr

class SecurityManager():
    def __init__(self, path):
        self.path = path
        self.base_data_dir = "./data"

    def validate_path(self):
        """Validates the path entered"""

        # 1. Checks if the path actually exists
        if not self.path or not isinstance(self.path, str):
            raise ValueError("Path must be a non-empty string")
        
        # Alters the path into its resolute form
        try:
            resolved = Path(self.path).expanduser().resolve(strict = False)
        except Exception as e:
            raise ValueError(f"Invalid Path: {e}")
        
        resolved_str = str(resolved)

        # 2. Resolves the allowed base, checks if the path is inside the allowed base
        allowed_base = Path(self.base_data_dir).resolve()
        if not resolved_str.startswith(str(allowed_base) + os.sep) and resolved != allowed_base:
            raise ValueError(f"Path must be inside the allowed directory: {self.base_data_dir}")
        
        # Creates the train and test directory according to path structure
        train_dir = resolved / "train" 
        test_dir = resolved / "test"

        # 3. Checks if train directory exists and is a directory
        if not train_dir.exists() or not train_dir.is_dir():
            raise ValueError("Train directory must exist inside the dataset folder")

        # 4. Checks if test directory exists and is a directory
        if not test_dir.exists() or not test_dir.is_dir():
            raise ValueError("Test directory must exist inside the dataset folder")
        
        # List comprehension used to get the class names from the train and test directory by iterating through it 
        # and checking if it is a directory
        train_classes = [d for d in train_dir.iterdir() if d.is_dir()]
        test_classes = [d for d in test_dir.iterdir() if d.is_dir()]

        # 5. If there are less than two classes in either train or test folder, raise an error
        if len(train_classes) | len(test_classes) < 2:
            raise ValueError("Train and Test folder must have greater than 2 classes")

        # Get the actual class names in the train and test directory through list comprehension and sorting
        train_class_names = sorted([d.name for d in train_classes])
        test_class_names = sorted([d.name for d in test_classes])

        # 6. If the class names in train folder do not match test folder, raise an error
        if train_class_names != test_class_names:
            raise ValueError("Train and test folder must have the same classes")
        
        # Iterates through every file in the train and test folder and stores it in images
        for class_dir in train_classes + test_classes:
            images = list(class_dir.glob("*"))
        
        # Function verifies if it is an image or not
        def verify_image(path):
            image_type = imghdr.what(path)

            if image_type:
                return True
            else:
                return False
        
        # Stores all files which are not images
        invalid_images = [f for f in images if not verify_image(f)]

        # 7. If there are one or more files which are not images, raise an error
        if len(invalid_images) > 0:
            raise ValueError(f"Train or test folder contain one or more files that are not an image: {invalid_images}")
        # Return true if the path has been validated and no errors have been raised
        return True
           