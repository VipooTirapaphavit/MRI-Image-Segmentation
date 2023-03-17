import os
from glob import glob
import torch
from monai.transforms import(
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
)

from monai.data import DataLoader, Dataset
from monai.utils import first
import matplotlib.pyplot as plt


data_dir = 'D:/Liver Segmentation/Kaggle/nifti_files'
train_images = sorted(glob(os.path.join(data_dir, "TrainImages", "*.nii.gz")))
train_labels = sorted(glob(os.path.join(data_dir, "TrainLabels", "*.nii.gz")))

val_images = sorted(glob(os.path.join(data_dir, "ValImages", "*.nii.gz")))
val_labels = sorted(glob(os.path.join(data_dir, "ValLabels", "*.nii.gz")))

train_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
test_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(val_images, val_labels)]

orig_transforms = Compose(
    
    [
     LoadImaged(keys =['image', 'label']),
     AddChanneld(keys =['image', 'label']),
     ToTensord(keys =['image', 'label'])
    ]
)

train_transforms = Compose(
    
    [
     LoadImaged(keys =['image', 'label']),
     AddChanneld(keys =['image', 'label']),
     Spacingd(keys =['image', 'label'], pixdim = (1.5,1.5,2)),
     ScaleIntensityRanged(keys ='image', a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True),
     CropForegroundd(keys =['image', 'label'], source_key='image'),
     Resized(keys =['image', 'label'], spatial_size = [128,128,128]),
     ToTensord(keys =['image', 'label'])
    ]
)

val_transforms = Compose(
    
    [
     LoadImaged(keys =['image', 'label']),
     AddChanneld(keys =['image', 'label']),
     Spacingd(keys =['image', 'label'], pixdim = (1.5,1.5,2)),
     ScaleIntensityRanged(keys ='image', a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True),
     ToTensord(keys =['image', 'label'])
    ]
)

orig_ds = Dataset(data = train_files, transform = orig_transforms)
orig_loader = DataLoader(orig_ds, batch_size = 1)

train_ds = Dataset(data = train_files, transform = train_transforms)
train_loader = DataLoader(train_ds, batch_size = 1)

val_ds = Dataset(data = test_files, transform = val_transforms)
val_loader = DataLoader(val_ds, batch_size = 1)


test_patient = first(train_loader)
orig_patient = first(orig_loader)


plt.figure('test', (12,6))

plt.subplot(1,3,1)
plt.title('Orig patient')
plt.imshow(orig_patient['image'][0,0,:,:,30], cmap='gray')

plt.subplot(1,3,2)
plt.title('Slice of a patient')
plt.imshow(test_patient['image'][0,0,:,:,30], cmap='gray')

plt.subplot(1,3,3)
plt.title('Label of a patient')
plt.imshow(test_patient['label'][0,0,:,:,30])
plt.show()