from modelhandler import ModelManager
from securityhandler import SecurityManager
from torchvision import transforms

training_transform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ColorJitter(
                                                                    brightness=0.2,
                                                                    contrast=0.2,
                                                                    saturation=0.2,
                                                                    hue=0.1
                                                                ),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                   std=[0.229, 0.224, 0.225]
                                                                   )])

testing_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]
                                        )])

sh = SecurityManager("/Users/RonenGupta/Desktop/HSCSoftwareEngineering_Task-3/hymenoptera_data")
if sh.validate_path():
    mm = ModelManager("/Users/RonenGupta/Desktop/HSCSoftwareEngineering_Task-3/hymenoptera_data/train", "/Users/RonenGupta/Desktop/HSCSoftwareEngineering_Task-3/hymenoptera_data/test", 10, training_transform, testing_transform, 1e-4, 32, 64, True)
    train_dataloader = mm.train_transforms_dataset()
    test_dataloader = mm.test_transforms_dataset()
    model = mm.build()
    mm.train()
    mm.test()



