# MultiheadEfficientVit
multihead model for lane line and drivable area segmentation based on efficientVit model

Autonomous driving requires a robust and efficient perception system to accurately detect lane lines and identify drivable areas, which are crucial for safe navigation. We introduce MultiEfficientViT, a novel multi-task model designed to perform lane line segmentation and drivable area segmentation concurrently. MultiEfficientViT leverages a single transformer-based encoder for feature extraction, combined with two decoders optimized for each specific task. Our model demonstrates superior performance on the challenging BDD100K dataset, setting new benchmarks for accuracy and speed in both tasks.
To further refine MultiEfficientViT, we conducted fine-tuning on a custom local dataset tailored to specific driving environments. This fine-tuning process allows the model to adapt to unique road conditions and improve its generalization capabilities. To support this, we introduce a new dataset specifically created for lane line and drivable area segmentation in varied driving scenarios. The fine-tuning was executed using Kaggle's P100 GPUs, enabling efficient training and optimization.
Our extensive experiments show that MultiEfficientViT not only excels on the BDD100K dataset but also adapts effectively to new environments through fine-tuning. This makes it a versatile solution for autonomous driving perception tasks, providing both high accuracy and real-time processing capabilities. 

![model0](https://github.com/user-attachments/assets/cae28173-a8a2-41e0-9741-de974363788e)
BDD100k example:

![image](https://github.com/user-attachments/assets/8d65e6a7-433b-4e80-addf-5ce90dfdeb75)

IADD dataset examples:

![image](https://github.com/user-attachments/assets/5696f490-ee84-426d-a3bb-1bcdf19c7c19)

Our source code is inspired by: EfficientVit , Twinlitenet
