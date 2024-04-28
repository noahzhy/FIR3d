# Fall recognition with a low-resolution thermal camera

> April 28, 2024  
@Haoyu
> 

## Abstract

As the aging population grows, the need for efficient fall detection systems becomes increasingly crucial for ensuring the well-being of the elderly.

We record a dataset of xxx video clips of falls and non-falls using a low-resolution thermal camera. We also generate a synthetic dataset of xxx fall and non-fall sequences using a physics-based simulator and propose a lightweight and real-time model to detect falls using low-resolution thermal cameras in eldercare facilities. 

Our model achieves an accuracy of xx% on a dataset of xxx fall and non-fall sequences, outperforming existing methods by a significant margin.

## 1. Introduction

Falls are a leading cause of injury and death among the elderly population. According to the Centers for Disease Control and Prevention, millions of older adults fall each year, resulting in significant healthcare costs and reduced quality of life. Many falls go unnoticed, especially in eldercare facilities where staff may not be present at all times. It's worth noting that this issue extends beyond eldercare facilities; even in the postoperative care of otherwise healthy individuals, falls remain a concern. Therefore, addressing fall prevention comprehensively is essential across all healthcare settings.

To address this issue, researchers have developed various fall detection systems using different sensors, such as accelerometers, gyroscopes, and cameras. However, most existing systems rely on high-resolution RGB cameras, which may raise privacy concerns in eldercare facilities and other healthcare settings. Furthermore, the camera based systems may not work well in low-light and no-light conditions, limiting their effectiveness in real-world scenarios.

In this work, we propose a lightweight and real-time fall detection model using a low-resolution thermal camera. Thermal cameras offer several advantages over RGB cameras, including the ability to work in no-light conditions, the ability to detect falls based on body heat signatures, and the preservation of privacy by utilizing low-resolution thermal imaging, we mitigate concerns about intrusive surveillance, ensuring the dignity and autonomy of the elderly.

and low-cost MLX90640 thermal camera from Melexis. Considering covering as much area of the room as possible, we chose the MLX90640-D110 model sensor to provide a wider field of view.

## 2. Related Work

Action recognition and fall detection have been extensively studied in the computer vision and machine learning communities. Most existing methods rely on RGB cameras, depth sensors, or wearable devices to capture human motion and detect falls. For example,

Fall detection using RGB cameras: Several methods have been proposed to detect falls using RGB cameras. These methods typically involve background subtraction, motion detection, and human pose estimation to identify falls. However, RGB cameras may raise privacy concerns in eldercare facilities and other healthcare settings.



## 3. Approach
### 3.1 Dataset


### 3.2 Model Architecture


### 3.3 Loss Function


### 3.4 Training Details


## 4. Experiments
### 4.1 Evaluation Metrics

## 5. Conclusion

## References

```

